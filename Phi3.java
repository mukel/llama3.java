///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector

// Practical Phi 3 mini 4k inference in a single Java file
// This file uses Alfonso² Peterssen's Llama3.java.
// Adaption to to Phi 3: Sascha Rogmann
//
// Supports llama.cpp's GGUF format, restricted to Q4_0 and Q8_0 quantized models of phi-3-mini-4k.
// Multi-threaded matrix vector multiplication routines implemented using Java's Vector API
// Simple CLI with --chat and --instruct mode
//
// To run just:
// javac --enable-preview --add-modules=jdk.incubator.vector Llama3.java Phi3.java
// java --enable-preview --add-modules=jdk.incubator.vector Phi3 --help
// # Download Q8_0 of https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF gently or quantize phi-3-mini-4k to Q4_0.
// java --enable-preview --add-modules=jdk.incubator.vector Phi3 --model /your_path/Phi-3.1-mini-4k-instruct-Q8_0.gguf --prompt "How to write 'three little cats' in chinese? Add an emoji."
//
// Enjoy!

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Implementation of the phi-3 based model Phi-3-mini-4k-instruct.
 * 
 * <p>This class makes use of the classes in Llama3.java.</p>
 * <ul>
 * <li>https://github.com/ggerganov/llama.cpp/</li>
 * <li>https://github.com/huggingface/transformers/tree/main/src/transformers/models/phi3</li>
 * </ul>
 */
public class Phi3 {
    final class Phi3ModelLoader {
        private static final String TOKENIZER_LLAMA_MODEL = "llama";
        
        /** Special token "&lt;s&gt;" */
        private static String TOKEN_BOS = "<s>";
        /** id of token "&lt;s&gt;" */
        private static int TOKEN_BOS_ID = 1;

        private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

        private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
            String model = (String) metadata.get("tokenizer.ggml.model");
            if (!TOKENIZER_LLAMA_MODEL.equals(model)) {
                throw new IllegalArgumentException("expected " + TOKENIZER_LLAMA_MODEL + " but found " + model);
            }
            String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
            return new Vocabulary(tokens, null);
        }

        public static Phi3Model loadModel(Path ggufPath, int contextLength) throws IOException {
            final String modelPrefix = "phi3.";
            try (var ignored = Timer.log("Load Phi3-model")) {
                GGUF gguf = GGUF.loadModel(ggufPath);
                Map<String, Object> metadata = gguf.getMetadata();

                Vocabulary vocabulary = loadVocabulary(metadata);
                Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

                int modelContextLength = (int) metadata.get(modelPrefix + "context_length");
                if (contextLength < 0 || modelContextLength < contextLength) {
                    contextLength = modelContextLength;
                }

                Llama.Configuration config = new Llama.Configuration(
                        (int) metadata.get(modelPrefix + "embedding_length"),
                        (int) metadata.get(modelPrefix + "feed_forward_length"),
                        (int) metadata.get(modelPrefix + "block_count"),
                        (int) metadata.get(modelPrefix + "attention.head_count"),

                        metadata.containsKey(modelPrefix + "attention.head_count_kv")
                                ? (int) metadata.get(modelPrefix + "attention.head_count_kv")
                                : (int) metadata.get(modelPrefix + "attention.head_count"),

                        vocabulary.size(),
                        contextLength,
                        false,
                        (float) metadata.getOrDefault(modelPrefix + "attention.layer_norm_rms_epsilon", 1e-5f),
                        (float) metadata.getOrDefault(modelPrefix + "rope.freq_base", 10000f)
                );

                Map<String, GGMLTensorEntry> tensorEntries = gguf.getTensorEntries();

                Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta);
                float[] ropeFreqsReal = ropeFreqs.first();
                float[] ropeFreqsImag = ropeFreqs.second();

                // Llama3: attn_k, attn_q, attn_v, ffn_down, ffn_up, ffn_norm, ffn_gate
                // Phi3: attn_qkv, ffn_down, ffn_up, ffn_norm
                Phi3.Phi3Model.Weights qw = new Phi3.Phi3Model.Weights(
                        ModelLoader.loadQuantized(tensorEntries.get("token_embd.weight")),
                        ModelLoader.loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                        ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),
                        ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                        ModelLoader.loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                        ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                        ModelLoader.loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                        ModelLoader.toFloatBuffer(tensorEntries.get("output_norm.weight")),
                        FloatBuffer.wrap(ropeFreqsReal),
                        FloatBuffer.wrap(ropeFreqsImag),
                        ModelLoader.loadQuantized(tensorEntries.get("output.weight"))
                );

                return new Phi3Model(config, tokenizer, qw);
            }
        }

        private static Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
            List<Pair<Integer, Integer>> merges = Collections.emptyList();

            int allTokens = vocabulary.size();
            int baseTokens = 32000; // assume all tokens after the base ones are special.
            //int reservedSpecialTokens = allTokens - baseTokens;
            List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

            assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

            Map<String, Integer> specialTokens =
                    IntStream.range(0, specialTokensList.size())
                            .boxed()
                            .collect(Collectors.toMap(
                                    i -> specialTokensList.get(i),
                                    i -> baseTokens + i)
                            );
            specialTokens.put(TOKEN_BOS, TOKEN_BOS_ID);

            return new TokenizerSPM(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
        }

    }

    record Phi3Model(Llama.Configuration configuration, Tokenizer tokenizer, Weights weights) {
        public State createNewState() {
            State state = new State(configuration());
            state.latestToken = tokenizer.getSpecialTokens().get("<s>");
            return state;
        }

        public static final class Weights {
            // token embedding table
            public final FloatTensor token_embedding_table; // (vocab_size, dim)
            // weights for rmsnorms
            public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
            // weights for matmuls
            // Llama3 q(layer, n_heads * head_size), Llama3 k (layer, n_kv_heads, head_size), Llama3 v (layer, n_kv_heads * head_size)
            // Phi3: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
            public final FloatTensor[] wqkv; // nn.Linear(hidden_size, op_size, bias=False)
            // Phi3: o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)
            public final FloatBuffer[] rms_ffn_weight; // (layer, dim)
            // weights for ffn
            public final FloatTensor[] wDown; // ffn_down, (layer, dim, hidden_dim)
            public final FloatTensor[] wGateUp; // ffn_up, (layer, hidden_dim, 2 * dim)
            // public final rmsnorm
            public final FloatBuffer rms_final_weight; // (dim,)
            // freq_cis for RoPE relatively positional embeddings
            public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
            public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
            // (optional) classifier weights for the logits, on the last layer
            public final FloatTensor wcls; // (vocab_size, dim)

            public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wqkv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight, FloatTensor[] wDown, FloatTensor[] wGateUp, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
                this.token_embedding_table = token_embedding_table;
                this.rms_att_weight = rms_att_weight;
                this.wqkv = wqkv;
                this.wo = wo;
                this.rms_ffn_weight = rms_ffn_weight;
                this.wDown = wDown;
                this.wGateUp = wGateUp;
                this.rms_final_weight = rms_final_weight;
                this.freq_cis_real = freq_cis_real;
                this.freq_cis_imag = freq_cis_imag;
                this.wcls = wcls;
            }
        }

        public static final class State {

            // current wave of activations
            public final FloatTensor x; // activation at current time stamp (dim,)
            public final FloatTensor xb; // same, but inside a residual branch (dim,)
            public final FloatTensor xb2; // an additional buffer just for convenience (dim,)
            public final FloatTensor hb; // buffer for hidden dimension in the ffn (2 * hidden_dim,)
            public final FloatTensor hbG; // mlp_gate, buffer for hidden dimension in the ffn (hidden_dim,)
            public final FloatTensor hbU; // mlp_up, buffer for hidden dimension in the ffn (hidden_dim,)
            public final FloatTensor qkv; // query-key-value (opSize,)
            public final FloatTensor q; // query-key-value (dim,)
            public final FloatTensor k; // query-key-value (nKVHeads * headDim,)
            public final FloatTensor v; // query-key-value (nKVHeads * headDim,)
            public final FloatTensor att; // buffer for scores/attention values (n_heads, seq_len)
            public final FloatTensor logits; // output logits
            // kv cache
            public final FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
            public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)

            public int latestToken;

            State(Llama.Configuration config) {
                this.x = ArrayFloatTensor.allocate(config.dim);
                this.xb = ArrayFloatTensor.allocate(config.dim);
                this.xb2 = ArrayFloatTensor.allocate(config.dim);
                this.hb = ArrayFloatTensor.allocate(2 * config.hiddenDim);
                this.hbG = ArrayFloatTensor.allocate(config.hiddenDim);
                this.hbU = ArrayFloatTensor.allocate(config.hiddenDim);
                final int opSize = config.dim + 2 * (config.numberOfKeyValueHeads * config.headSize);
                this.qkv = ArrayFloatTensor.allocate(opSize);
                this.q = ArrayFloatTensor.allocate(config.dim);
                final int headDim = config.dim / config.numberOfHeads;
                this.k = ArrayFloatTensor.allocate(config.numberOfKeyValueHeads * headDim);
                this.v = ArrayFloatTensor.allocate(config.numberOfKeyValueHeads * headDim);
                this.att = ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength);
                this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
                int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
                this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
                this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            }
        }

        static FloatTensor forward(Phi3Model model, Phi3Model.State state, int token, int position) {
            // a few convenience variables
            Llama.Configuration config = model.configuration();
            Phi3Model.Weights weights = model.weights();
            int dim = config.dim;
            int headSize = config.headSize;
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
            float sqrtHeadSize = (float) Math.sqrt(headSize);
            // dim=3072, headSize=96, kvDim=3072, kvMul=1
            // System.out.println(String.format("dim=%d, headSize=%d, kvDim=%d, kvMul=%d", dim, headSize, kvDim, kvMul));

            // copy the token embedding into x
            weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

            boolean debug = false;
            if (debug) {
                System.out.println(String.format("Embedding: %s, ..., %s",
                        IntStream.range(0, 3).mapToObj(i -> Float.toString(state.x.getFloat(i))).collect(Collectors.joining(", ")),
                        IntStream.range(dim - 3, dim).mapToObj(i -> Float.toString(state.x.getFloat(i))).collect(Collectors.joining(", "))));
            }

            // Phi3: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
            final int opSize = dim + 2 * (config.numberOfKeyValueHeads * headSize);
            if (debug) {
                System.out.println("opSize = " + opSize);
                System.out.println(String.format("dim=%d, headSize=%d, nKVH=%d", dim, headSize, config.numberOfKeyValueHeads));
            }

            // forward all the layers
            for (int l = 0; l < config.numberOfLayers; l++) {
                // attention rmsnorm
                Llama.rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);
                
                if (debug && (l < 10 || l > config.numberOfLayers - 3)) {
                    System.out.println(String.format("Layer %d: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.xb.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(dim - 3, dim).mapToObj(i -> Float.toString(state.xb.getFloat(i))).collect(Collectors.joining(", "))));
                }

                // qkv matmuls for this position
                // wqkv: (hidden_size, op_size)
                weights.wqkv[l].matmul(state.xb, state.qkv, opSize, dim);
                if (debug && l < 3) {
                    System.out.println(String.format("Layer %d, wqkv: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.qkv.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(opSize - 3, opSize).mapToObj(i -> Float.toString(state.qkv.getFloat(i))).collect(Collectors.joining(", "))));
                }
                // query_pos = self.num_heads * self.head_dim
                // query_states = qkv[..., :query_pos]
                state.qkv.copyTo(0, state.q, 0, dim);
                // key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
                state.qkv.copyTo(dim, state.k, 0, config.numberOfKeyValueHeads * headSize);
                // value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
                state.qkv.copyTo(dim + config.numberOfKeyValueHeads * headSize,
                        state.v, 0, config.numberOfKeyValueHeads * headSize);
                if (debug && l < 3) {
                    System.out.println(String.format("Layer %d, before q.RoPE: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.q.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(dim - 3, dim).mapToObj(i -> Float.toString(state.q.getFloat(i))).collect(Collectors.joining(", "))));
                    System.out.println(String.format("Layer %d, before k.RoPE: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.k.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(dim - 3, dim).mapToObj(i -> Float.toString(state.k.getFloat(i))).collect(Collectors.joining(", "))));
                }

                // RoPE relative positional encoding: complex-valued rotate q and k in each head
                // phi-3 uses RoPE-type neox, i.e. offset dim/2 instead of 1.
                int dimHalf = headSize / 2;
                for (int i = 0; i < dim; i += 2) {
                    int head_dim = i % headSize;
                    int base = i - head_dim;
                    int ic = base + head_dim / 2;
                    float fcr = weights.freq_cis_real.get(position * (headSize / 2) + (head_dim / 2));
                    float fci = weights.freq_cis_imag.get(position * (headSize / 2) + (head_dim / 2));
                    int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    for (int v = 0; v < rotn; v++) {
                        FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(ic);
                        float v1 = vec.getFloat(ic + dimHalf);
                        vec.setFloat(ic, v0 * fcr - v1 * fci);
                        vec.setFloat(ic + dimHalf, v0 * fci + v1 * fcr);
                        if (debug && l < 3 && ic < 3) {
                            System.out.println(String.format("rope fwd: ic=%d, ic2=%d, v=%d, v0=%f, v1=%f, fcr=%f, fci=%f, dst0=%f, dsth=%f",
                                    ic, ic + dimHalf, v, v0, v1, fcr, fci, vec.getFloat(ic), vec.getFloat(ic + dimHalf)));                            
                        }
                    }
                }
                if (debug && l < 3) {
                    System.out.println(String.format("Layer %d, q.RoPE: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.q.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(dim - 3, dim).mapToObj(i -> Float.toString(state.q.getFloat(i))).collect(Collectors.joining(", "))));
                    System.out.println(String.format("Layer %d, k.RoPE: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.k.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(dim - 3, dim).mapToObj(i -> Float.toString(state.k.getFloat(i))).collect(Collectors.joining(", "))));
                }

                // save key,value at this time step (position) to our kv cache
                //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
                state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
                state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

                int curLayer = l;

                // multihead attention. iterate over all heads
                final int idxLayer = l;
                Parallel.parallelFor(0, config.numberOfHeads, h -> {
                    // get the query vector for this head
                    // float* q = s.q + h * headSize;
                    int qOffset = h * headSize;

                    // attention scores for this head
                    // float* att = s.att + h * config.seq_len;
                    int attOffset = h * config.contextLength;

                    // iterate over all timesteps, including the current one
                    for (int t = 0; t <= position; t++) {
                        // get the key vector for this head and at this timestep
                        // float* k = s.key_cache + loff + t * dim + h * headSize;
                        int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                        // calculate the attention score as the dot product of q and k
                        float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                        score /= sqrtHeadSize;
                        // save the score to the attention buffer
                        state.att.setFloat(attOffset + t, score);
                    }

                    if (debug && h <= 2 && idxLayer < 3) {
                        System.out.println(String.format("Layer %d, Head %d, Attention: %s, ...", idxLayer, h,
                                IntStream.range(0, 3).mapToObj(i -> Float.toString(state.att.getFloat(attOffset + i))).collect(Collectors.joining(", "))));
                    }

                    // softmax the scores to get attention weights, from 0..position inclusively
                    state.att.softmaxInPlace(attOffset, position + 1);

                    if (debug && h <= 2 && idxLayer < 3) {
                        System.out.println(String.format("Layer %d, Head %d, Attention: %s, ...", idxLayer, h,
                                IntStream.range(0, 3).mapToObj(i -> Float.toString(state.att.getFloat(attOffset + i))).collect(Collectors.joining(", "))));
                    }

                    // weighted sum of the values, store back into xb
                    // float* xb = s.xb + h * headSize;
                    int xbOffset = h * headSize;
                    // memset(xb, 0, headSize * sizeof(float));
                    state.xb.fillInPlace(xbOffset, headSize, 0f);

                    for (int t = 0; t <= position; t++) {
                        // get the value vector for this head and at this timestep
                        // float* v = s.value_cache + loff + t * dim + h * headSize;
                        int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                        // get the attention weight for this timestep
                        float a = state.att.getFloat(attOffset + t);
                        // accumulate the weighted value into xb
                        state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                    }
                });

                // final matmul to get the output of the attention
                weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

                // residual connection back into x
                state.x.addInPlace(state.xb2);

                // ffn rmsnorm
                Llama.rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);

                // MLP in phi3:
                // up_states = self.gate_up_proj(hidden_states)
                weights.wGateUp[l].matmul(state.xb, state.hb, 2 * config.hiddenDim, dim);
                // gate, up_states = up_states.chunk(2, dim=-1)
                copyChunk(state.hb, state.hbG, 2 * config.hiddenDim, config.hiddenDim, 2, 0);
                copyChunk(state.hb, state.hbU, 2 * config.hiddenDim, config.hiddenDim, 2, 1);
                if (debug && l < 3) {
                    System.out.println(String.format("Layer %d, mlpGateUp: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.hb.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(2 * config.hiddenDim - 3, 2 * config.hiddenDim).mapToObj(i -> Float.toString(state.hb.getFloat(i))).collect(Collectors.joining(", "))));
                    System.out.println(String.format("Layer %d, mlpGate: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.hbG.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(config.hiddenDim - 3, config.hiddenDim).mapToObj(i -> Float.toString(state.hbG.getFloat(i))).collect(Collectors.joining(", "))));
                    System.out.println(String.format("Layer %d, mlpUp: %s, ..., %s", l,
                            IntStream.range(0, 3).mapToObj(i -> Float.toString(state.hbU.getFloat(i))).collect(Collectors.joining(", ")),
                            IntStream.range(config.hiddenDim - 3, config.hiddenDim).mapToObj(i -> Float.toString(state.hbU.getFloat(i))).collect(Collectors.joining(", "))));
                }

                // self.activation_fn(gate)
                // SwiGLU non-linearity
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                state.hbG.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

                // up_states = up_states * self.activation_fn(gate)
                // elementwise multiply with w3(x)
                state.hbU.multiplyInPlace(state.hbG);
                
                // self.down_proj(up_states)
                weights.wDown[l].matmul(state.hbU, state.xb, dim, config.hiddenDim);

                // residual connection
                state.x.addInPlace(state.xb);
                
            }

            // final rmsnorm
            Llama.rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);

            // classifier into logits
            weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);

            return state.logits;
        }

        static void copyChunk(FloatTensor in, FloatTensor out, int dim1In, int dim1Out, int nChunks, int chunkNo) {
            assert(dim1In == dim1Out * nChunks);
            final int startOffsetInDim1 = chunkNo * dim1Out;
            Parallel.parallelFor(0, dim1Out, i -> {
                out.setFloat(i, in.getFloat(startOffsetInDim1 + i));
            });
        }


        /**
         * LLM generation entry point, ingest prompt tokens and generates new tokens.
         *
         * <p>
         * All prompt tokens are ingested first, then inference starts, until a stop token is found.
         * The returned tokens only include generated/inferred tokens.
         *
         * @param model            model to run inference (including weights, configuration, tokenizer ...)
         * @param state            state of the model e.g. key/value caches ... this is mutated by this call
         * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
         * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
         * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
         * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
         *                         if this value is negative or greater than {@link Configuration#contextLength context length}
         * @param sampler          {@link Sampler strategy} used to select tokens
         * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
         * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
         * @param finishOutput callback, if non-null, to finish the output if it ends with UTF-8-tokens 
         * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
         */
        public static List<Integer> generateTokens(Phi3Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                                   IntConsumer onTokenGenerated, Runnable finishOutput) {
            long startNanos = System.nanoTime();
            if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
                maxTokens = model.configuration().contextLength;
            }
            List<Integer> generatedTokens = new ArrayList<>(maxTokens);
            int token = state.latestToken; // BOS?
            int nextToken;
            int promptIndex = 0;
            ByteArrayOutputStream baos = new ByteArrayOutputStream(5);
            for (int position = startPosition; position < maxTokens; ++position) {
                Phi3Model.forward(model, state, token, position);
                if (promptIndex < promptTokens.size()) {
                    // Force-pick token from prompt.
                    nextToken = promptTokens.get(promptIndex++);
                    if (echo) {
                        // log prompt token (different color?)
                        System.out.println("NextToken: " + nextToken);
                        //System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                        String decoded = model.tokenizer.decodeImpl(List.of(nextToken));
                        System.err.print(decode(decoded, baos));
                    }
                } else {
                    nextToken = sampler.sampleToken(state.logits);
                    if (echo) {
                        // log inferred token
                        System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decodeImpl(List.of(nextToken))));
                    }
                    generatedTokens.add(nextToken);
                    if (onTokenGenerated != null) {
                        onTokenGenerated.accept(nextToken);
                    }
                    if (stopTokens.contains(nextToken)) {
                        break;
                    }
                }
                state.latestToken = token = nextToken;
                if (position == 2000) {
                    break;
                }
            }
            if (finishOutput != null) {
                finishOutput.run();
            }

            long elapsedNanos = System.nanoTime() - startNanos;
            int totalTokens = promptIndex + generatedTokens.size();
            System.err.printf("%n%.2f tokens/s (%d)%n", totalTokens / (elapsedNanos / 1_000_000_000.0), totalTokens);

            return generatedTokens;
        }
    }

    /** SPM-based llama-tokenizer (SentencePiece) */
    static class TokenizerSPM extends Tokenizer {
        private static final String SPM_UNDERSCORE = "\u2581";
        private final Vocabulary vocabulary;

        public TokenizerSPM(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern,
                Map<String, Integer> specialTokens) {
            super(vocabulary, merges, regexPattern, specialTokens);
            this.vocabulary = vocabulary;
        }
        
        @Override
        public List<Integer> encodeAsList(String pText) {
            String text = pText.replace(" ", SPM_UNDERSCORE);
            text = pText.startsWith(SPM_UNDERSCORE) ? text : SPM_UNDERSCORE + text;
            final int textLen = text.length();
            
            final List<Integer> tokens = new ArrayList<>();
            final int vocSize = vocabulary.size();
            int offset = 0;
            while (offset < textLen) {
                String curVoc = null;
                int token = -1;
                for (int j = 0; j < vocSize; j++) {
                    final String voc = vocabulary.get(j);
                    if (text.startsWith(voc, offset)
                            && (curVoc == null || curVoc.length() < voc.length())) {
                        curVoc = voc;
                        token = j;
                    }
                }
                if (curVoc == null) {
                    // Try <0xE7>... of character or surrogate (emoji).
                    final int len = (offset + 1 < textLen) && Character.isHighSurrogate(text.charAt(offset)) ? 2 : 1; 
                    final byte[] bufUtf8 = text.substring(offset, offset + len).getBytes(StandardCharsets.UTF_8);
                    for (int i = 0; i < bufUtf8.length; i++) {
                        final String sHex = String.format("<0x%02x>", bufUtf8[i] & 0xff);
                        token = -1;
                        for (int j = 0; j < vocSize; j++) {
                            if (sHex.equalsIgnoreCase(vocabulary.get(j))) {
                                token = j;
                            }
                        }
                        if (token == -1) {
                            throw new RuntimeException(String.format("Can't tokenize text at offset %d (%c / (%d, sHex %s)), tokens = %s, text: %s",
                                    offset, text.charAt(offset), i, sHex, tokens, text));
                        }
                        tokens.add(token);
                    }
                    offset += len;
                    continue;
                }
                tokens.add(token);
                offset += curVoc.length();
            }
            return tokens;
        }

        @Override
        public String decode(List<Integer> tokens) {
            final StringBuilder sb = new StringBuilder();
            for (Integer token : tokens) {
                sb.append(vocabulary.get(token));
            }
            return sb.toString().replace(SPM_UNDERSCORE, " ");
        }
    }
    
    /**
     * Utility tailored for Llama 3 instruct prompt format.
     */
    static class ChatFormat {

        protected final Tokenizer tokenizer;
        protected final int end;

        public ChatFormat(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
            this.end = specialTokens.get("<|end|>");
        }

        public Tokenizer getTokenizer() {
            return tokenizer;
        }

        public Set<Integer> getStopTokens() {
            return Set.of(end);
        }

        public List<Integer> encodeHeader(ChatFormat.Message message) {
            List<Integer> tokens = new ArrayList<>();
            String tokenRole = "<|" + message.role().name() + "|>";
            final Integer idxSpecial = tokenizer.getSpecialTokens().get(tokenRole);
            if (idxSpecial != null) {
                tokens.add(idxSpecial);
            } else {
                tokens.addAll(this.tokenizer.encodeAsList(tokenRole));
            }
            //tokens.addAll(this.tokenizer.encodeAsList("\n"));
            return tokens;
        }

        public List<Integer> encodeMessage(ChatFormat.Message message) {
            List<Integer> tokens = this.encodeHeader(message);
            tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
            tokens.add(tokenizer.getSpecialTokens().get("<|end|>"));
            return tokens;
        }

        public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
            List<Integer> tokens = new ArrayList<>();
            //tokens.add(beginOfText);
            for (ChatFormat.Message message : dialog) {
                tokens.addAll(this.encodeMessage(message));
            }
            if (appendAssistantTurn) {
                // Add the start of an assistant message for the model to complete.
                tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            }
            return tokens;
        }

        public record Message(ChatFormat.Role role, String content) {
        }

        public record Role(String name) {
            public static ChatFormat.Role SYSTEM = new ChatFormat.Role("system");
            public static ChatFormat.Role USER = new ChatFormat.Role("user");
            public static ChatFormat.Role ASSISTANT = new ChatFormat.Role("assistant");

            @Override
            public String toString() {
                return name;
            }
        }
    }

    /** pattern to match UTF-8-tokens as &lt;0x0A&gt; */
    static Pattern P_UTF8_BYTE = Pattern.compile("<0x([0-9A-F]{2})>");
    
    static void runInstructOnce(Phi3Model model, Sampler sampler, Llama3.Options options) {
        Phi3.Phi3Model.State state = model.createNewState();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());
        System.out.println(String.format("JVM: %s / %s / %s",
                System.getProperty("java.vm.vendor"), System.getProperty("java.vm.name"), System.getProperty("java.vm.version")));
        System.out.println("Prompt: " + options.prompt());

        List<Integer> promptTokens = new ArrayList<>();
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        if (options.echo()) {
            System.out.println("Prompt tokens: " + promptTokens);
        }

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        List<Integer> responseTokens = Phi3Model.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            if (options.stream()) {
                if (!model.tokenizer().isSpecialToken(token)) {
                    String decoded = model.tokenizer.decodeImpl(List.of(token));
                    System.out.print(decode(decoded, baos));
                }
            }
        }, () -> System.out.print(decode("", baos)));
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            baos.reset();
            final String responseText = responseTokens.stream()
                    .map(token -> model.tokenizer.decodeImpl(List.of(token)))
                    .map(sToken -> decode(sToken, baos)).collect(Collectors.joining())
                + decode("", baos);
            System.out.println(responseText);
        }
    }

    /**
     * Replace decodedUtf8-tokens as &lt;0x0A&gt; by bytes.
     * Replace SPM-underscore by space.
     * @param decoded string of one token to be decoded
     * @param baos buffer to store the current c-8-sequence
     * @return decoded string
     */
    static String decode(String decoded, ByteArrayOutputStream baos) {
        String decodedUtf8 = decoded;
        Matcher mUtf8Seq = P_UTF8_BYTE.matcher(decoded);
        if (mUtf8Seq.matches()) {
            baos.write(Integer.parseInt(mUtf8Seq.group(1), 16));
            decodedUtf8 = "";
        } else {
            if (baos.size() > 0) {
                decodedUtf8 = new String(baos.toByteArray(), StandardCharsets.UTF_8) + decoded;
                baos.reset();
            }
        }
        return decodedUtf8.replace(TokenizerSPM.SPM_UNDERSCORE, " ");
    }

    static void runInteractive(Phi3Model model, Sampler sampler, Llama3.Options options) {
        Phi3.Phi3Model.State state = model.createNewState();
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            if (List.of("quit", "exit").contains(userText)) {
                break;
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            List<Integer> responseTokens = Phi3Model.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
                if (options.stream()) {
                    if (!model.tokenizer().isSpecialToken(token)) {
                        System.out.print(decode(model.tokenizer().decodeImpl(List.of(token)), baos));
                    }
                }
            }, () -> System.out.print(decode("", baos)));
            // Include stop token in the prompt history, but not in the response displayed to the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if (!options.stream()) {
                baos.reset();
                String responseText = responseTokens.stream()
                    .map(token -> model.tokenizer.decodeImpl(List.of(token)))
                    .map(sToken -> {System.out.println("Token: " + sToken);return decode(sToken, baos);})
                    .collect(Collectors.joining());
                System.out.println(responseText);
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        Llama3.Options options = Llama3.Options.parseOptions(args);
        Phi3Model model = Phi3ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        Sampler sampler = Llama3.selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }

}
