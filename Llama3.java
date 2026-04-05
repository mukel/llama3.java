///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 21+
//PREVIEW
//COMPILE_OPTIONS --add-modules=jdk.incubator.vector
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
//MAIN com.llama4j.Llama3

// Practical Llama 3 (and 3.1) inference in a single Java file
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
//
// Supports llama.cpp's GGUF format with Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16/F32 weights
// Multi-threaded matrix vector multiplication routines implemented using Java's Vector API
// Simple CLI with --chat and --instruct mode
//
// To run just:
// jbang Llama3.java --help
//
// Enjoy!
package com.llama4j;

import jdk.incubator.vector.*;
import sun.misc.Unsafe;

import java.io.IOException;
import java.io.PrintStream;
import java.io.BufferedInputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.LongConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;

public class Llama3 {
    // Batch-size used in prompt evaluation.
    private static final int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);

    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.ARGMAX;
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            if (topp <= 0 || topp >= 1) {
                // simply sample from the predicted probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }
            sampler = logits -> {
                // apply the temperature to the logits
                logits.divideInPlace(0, logits.size(), temperature);
                // apply softmax to the logits to get the probabilities for next token
                logits.softmaxInPlace(0, logits.size());
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    static void runInteractive(Llama model, Sampler sampler, Options options) {
        Llama.State state = null;
        List<Integer> conversationTokens = new ArrayList<>();
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());
        conversationTokens.add(chatFormat.beginOfText);
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        int startPosition = 0;
        Scanner in = new Scanner(System.in);
        loop: while (true) {
            System.out.print("> ");
            System.out.flush();
            String userText = in.nextLine();
            switch (userText) {
                case "/quit":
                case "/exit": break loop;
                case "/context": {
                    System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                            conversationTokens.size(),
                            options.maxTokens(),
                            options.maxTokens() - conversationTokens.size());
                    continue;
                }
            }
            if (state == null) {
                state = model.createNewState(BATCH_SIZE);
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens = Llama.generateTokens(model, state, startPosition, conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
                if (options.stream()) {
                    if (!model.tokenizer().isSpecialToken(token)) {
                        System.out.print(model.tokenizer().decode(List.of(token)));
                    }
                }
            });
            // Include stop token in the prompt history, but not in the response displayed to the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if (!options.stream()) {
                String responseText = model.tokenizer().decode(responseTokens);
                System.out.println(responseText);
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
        }
    }

    static void runInstructOnce(Llama model, Sampler sampler, Options options) {
        Llama.State state = model.createNewState(BATCH_SIZE);
        ChatFormat chatFormat = new ChatFormat(model.tokenizer());

        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.beginOfText);
        if (options.systemPrompt() != null) {
            promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.systemPrompt())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.prompt())));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), token -> {
            if (options.stream()) {
                if (!model.tokenizer().isSpecialToken(token)) {
                    System.out.print(model.tokenizer().decode(List.of(token)));
                }
            }
        });
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        if (!options.stream()) {
            String responseText = model.tokenizer().decode(responseTokens);
            System.out.println(responseText);
        }
    }

    record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
                   float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo) {

        static final int DEFAULT_MAX_TOKENS = 512;

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                System.out.println("ERROR " + messageFormat.formatted(args));
                System.out.println();
                printUsage(System.out);
                System.exit(-1);
            }
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  jbang Llama3.java [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --system-prompt, -sp <string> (optional) system prompt");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 0.1");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
            out.println();
            out.println("Examples:");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Tell me a joke\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Reply concisely, in French\" --prompt \"Who was Marie Curie?\"");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --system-prompt \"Answer concisely\" --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --chat");
            out.println("  jbang Llama3.java --model llama3.2-1b-q4_0.gguf --prompt \"Print 5 emojis\" --stream=false");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String systemPrompt = null;
            float temperature = 0.1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            // Keep max context length small for low-memory devices.
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false;
            boolean stream = true;
            boolean echo = false;

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--help", "-h" -> {
                        printUsage(System.out);
                        System.exit(0);
                    }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1; // skip arg
                        }
                        switch (optionName) {
                            case "--prompt", "-p" -> prompt = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Paths.get(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                            case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Llama model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            // No compatible preloaded model found, fallback to fully parse and load the specified file.
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
        }
        Sampler sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
        if (options.interactive()) {
            runInteractive(model, sampler, options);
        } else {
            runInstructOnce(model, sampler, options);
        }
    }
}

final class GGUF {
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    // Buffered read window for GGUF header + metadata + tensor-info parsing.
    // This avoids many tiny FileChannel.read() calls.
    private static final int PARSE_BUFFER_SIZE = 1 << 20;
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(3);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    private Map<String, GGUFTensorInfo> tensorInfos;

    private long tensorDataOffset;

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private long parsePosition;

    public static GGUF loadModel(FileChannel fileChannel, String modelLabel) throws IOException {
        try (var ignored = Timer.log("Parse " + modelLabel)) {
            // A caller may reuse an already-open channel after previous reads; always parse from start.
            fileChannel.position(0L);
            GGUF gguf = new GGUF();
            ReadableByteChannel channel = Channels.newChannel(
                    new BufferedInputStream(Channels.newInputStream(fileChannel), PARSE_BUFFER_SIZE)
            );
            gguf.parsePosition = 0L;
            gguf.loadModelImpl(channel);
            return gguf;
        }
    }

    enum MetadataValueType {
        // The value is a 8-bit unsigned integer.
        // (represented as signed byte in Java)
        UINT8(1),
        // The value is a 8-bit signed integer.
        INT8(1),
        // The value is a 16-bit unsigned little-endian integer.
        UINT16(2),
        // The value is a 16-bit signed little-endian integer.
        INT16(2),
        // The value is a 32-bit unsigned little-endian integer.
        UINT32(4),
        // The value is a 32-bit signed little-endian integer.
        INT32(4),
        // The value is a 32-bit IEEE754 floating point number.
        FLOAT32(4),
        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        // (0 = false, 1 = true)
        BOOL(1),
        // The value is a UTF-8 non-null-terminated string, with length prepended.
        // Variable-size payload (length-prefixed), therefore byteSize is marked as -8.
        STRING(-8),
        // The value is an array of other values, with element type and length prepended.
        // Arrays can be nested.
        // Variable-size payload, therefore byteSize is marked as -8.
        ARRAY(-8),
        // The value is a 64-bit unsigned little-endian integer.
        UINT64(8),
        // The value is a 64-bit signed little-endian integer.
        INT64(8),
        // The value is a 64-bit IEEE754 floating point number.
        FLOAT64(8);

        private final int byteSize;

        MetadataValueType(int byteSize) {
            this.byteSize = byteSize;
        }

        private static final MetadataValueType[] VALUES = values();

        public static MetadataValueType fromIndex(int index) {
            return VALUES[index];
        }

        public int byteSize() {
            return byteSize;
        }
    }

    private void loadModelImpl(ReadableByteChannel channel) throws IOException {
        // The header of the file.
        readHeader(channel); // gguf_header_t header;

        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUF.GGUFTensorInfo ti = readTensorInfo(channel);
            assert !tensorInfos.containsKey(ti.name);
            tensorInfos.put(ti.name, ti);
        }

        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        long position = parsePosition;
        int padding = (int) ((getAlignment() - (position % getAlignment())) % getAlignment());
        skipBytes(channel, padding);

        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This data should be close
        // or identical to the data in the original model file, but may be different due to quantization or
        // other optimizations for inference. Any such deviations should be recorded in the metadata or as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        this.tensorDataOffset = parsePosition;
    }

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        // Global arena keeps tensor mmap slices alive for the process lifetime.
        // (Arena.ofAuto() can release too early when references outlive parsing scope.)
        Arena arena = Arena.global();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            long numberOfElements = FloatTensor.numberOfElementsLong(ti.dimensions());
            long sizeInBytes = ti.ggmlType().byteSizeFor(numberOfElements);
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
    }

    private GGMLType readGGMLType(ReadableByteChannel channel) throws IOException {
        int ggmlTypeId = readInt(channel);
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUF.GGUFTensorInfo readTensorInfo(ReadableByteChannel channel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        // gguf_string_t name;
        String name = readString(channel);
        assert name.length() <= 64;

        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        // uint32_t n_dimensions;
        int n_dimensions = readInt(channel);
        assert n_dimensions <= 4;

        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(channel));
        }

        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(channel); // ggml_type type;

        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // The offset is relative to the tensor-data section, not file start.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(channel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUF.GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(ReadableByteChannel channel) throws IOException {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(channel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        // char string[len];
        return new String(readBytes(channel, len), StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(ReadableByteChannel channel) throws IOException {
        // The key of the metadata. It is a standard GGUF string, with the following caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        // The key of the metadata. It must be hierarchical lower_snake_case segments separated by '.'.
        String key = readString(channel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(channel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(ReadableByteChannel channel) throws IOException {
        // The type of the value.
        // Must be one of the gguf_metadata_value_type values.
        MetadataValueType valueType = readMetadataValueType(channel); // gguf_metadata_value_type value_type;

        // The value payload itself.
        return readMetadataValueOfType(valueType, channel); // gguf_metadata_value_t value;
    }

    void readHeader(ReadableByteChannel channel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        // Some readers may check for 0x46554747 on little-endian machines.
        this.magic = readInt(channel); // uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }

        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        // Must be 3 for the latest format in the GGUF spec.
        // This version should only be increased for structural file-format changes.
        this.version = readInt(channel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }

        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        // This is explicit to make tensor loading possible without metadata lookup.
        this.tensorCount = Math.toIntExact(readLong(channel)); // uint64_t tensor_count;

        // The number of metadata key-value pairs.
        this.metadata_kv_count = Math.toIntExact(readLong(channel)); // uint64_t metadata_kv_count;

        // The metadata key-value pairs.
        this.metadata = HashMap.newHashMap(metadata_kv_count); // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(channel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(ReadableByteChannel channel) throws IOException {
        // Any value type is valid, including arrays.
        // Any metadata value type is valid, including nested arrays.
        MetadataValueType valueType = readMetadataValueType(channel); // gguf_metadata_value_type type;

        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(channel)); // uint64_t len;

        // The array of values.
        // gguf_metadata_value_t array[len];
        switch (valueType) {
            case UINT8, INT8 -> {
                return readBytes(channel, len);
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(channel);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(channel);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(channel);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(channel);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(channel);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(channel);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + valueType);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, ReadableByteChannel channel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(channel);
            case UINT16, INT16 -> readShort(channel);
            case UINT32, INT32 -> readInt(channel);
            case FLOAT32 -> readFloat(channel);
            case UINT64, INT64 -> readLong(channel);
            case FLOAT64 -> readDouble(channel);
            case BOOL -> readBoolean(channel);
            case STRING -> readString(channel);
            case ARRAY -> readArray(channel);
        };
    }

    private MetadataValueType readMetadataValueType(ReadableByteChannel channel) throws IOException {
        int index = readInt(channel);
        return MetadataValueType.fromIndex(index);
    }

    private byte[] readBytes(ReadableByteChannel channel, int length) throws IOException {
        byte[] bytes = new byte[length];
        readFully(channel, ByteBuffer.wrap(bytes));
        return bytes;
    }

    private void skipBytes(ReadableByteChannel channel, int length) throws IOException {
        int remaining = length;
        byte[] scratch = new byte[Math.min(length, 1 << 12)];
        while (remaining > 0) {
            int chunk = Math.min(remaining, scratch.length);
            readFully(channel, ByteBuffer.wrap(scratch, 0, chunk));
            remaining -= chunk;
        }
    }

    private void readFully(ReadableByteChannel channel, ByteBuffer byteBuffer) throws IOException {
        int expected = byteBuffer.remaining();
        while (byteBuffer.hasRemaining()) {
            int bytesRead = channel.read(byteBuffer);
            if (bytesRead < 0) {
                throw new IOException("Unexpected EOF while reading GGUF metadata");
            }
        }
        parsePosition += expected;
    }

    private byte readByte(ReadableByteChannel channel) throws IOException {
        BB_1.clear();
        readFully(channel, BB_1);
        return BB_1.get(0);
    }

    private boolean readBoolean(ReadableByteChannel channel) throws IOException {
        return readByte(channel) != 0;
    }

    private short readShort(ReadableByteChannel channel) throws IOException {
        BB_2.clear();
        readFully(channel, BB_2);
        return BB_2.getShort(0);
    }

    private int readInt(ReadableByteChannel channel) throws IOException {
        BB_4.clear();
        readFully(channel, BB_4);
        return BB_4.getInt(0);
    }

    private long readLong(ReadableByteChannel channel) throws IOException {
        BB_8.clear();
        readFully(channel, BB_8);
        return BB_8.getLong(0);
    }

    private float readFloat(ReadableByteChannel channel) throws IOException {
        return Float.intBitsToFloat(readInt(channel));
    }

    private double readDouble(ReadableByteChannel channel) throws IOException {
        return Double.longBitsToDouble(readLong(channel));
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}

interface Timer extends AutoCloseable {
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                System.err.println(label + ": "
                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                        + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}

final class ModelLoader {
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";

    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!TOKENIZER_LLAMA_3_MODEL.equals(model)) {
            throw new IllegalArgumentException("expected " + TOKENIZER_LLAMA_3_MODEL + " but found " + model);
        }
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }

    public static Llama loadModel(Path ggufPath, int contextLength, boolean loadWeights) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
            GGUF gguf = GGUF.loadModel(fileChannel, ggufPath.toString());
            return loadModel(fileChannel, gguf, contextLength, loadWeights);
        }
    }

    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) throws IOException {
        try (var ignored = Timer.log("Load LlaMa model")) {
            Map<String, Object> metadata = gguf.getMetadata();
            Vocabulary vocabulary = loadVocabulary(metadata);
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

            Llama.Configuration config = new Llama.Configuration(
                    (int) metadata.get("llama.embedding_length"),
                    (int) metadata.get("llama.feed_forward_length"),
                    (int) metadata.get("llama.block_count"),
                    (int) metadata.get("llama.attention.head_count"),

                    metadata.containsKey("llama.attention.head_count_kv")
                            ? (int) metadata.get("llama.attention.head_count_kv")
                            : (int) metadata.get("llama.attention.head_count"),

                    vocabulary.size(),
                    (int) metadata.get("llama.context_length"),
                    (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                    (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)
            ).withContextLength(contextLength);

            Llama.Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            return new Llama(config, tokenizer, weights);
        }
    }

    static Llama.Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        float scaleFactor = 8;
        float loFreqFactor = 1;
        float hiFreqFactor = 3;
        int oldContextLength = 8192;
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        Llama.Weights qw = new Llama.Weights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                toFloatBuffer(tensorEntries.get("output_norm.weight")),
                FloatBuffer.wrap(ropeFreqsReal),
                FloatBuffer.wrap(ropeFreqsImag),
                // If "output.weight" is not present then the embedding weights are tied/shared with the decoder.
                // This is commonly referred as "tie word embeddings".
                loadQuantized(tensorEntries.getOrDefault("output.weight", tokenEmbeddings))
        );

        return qw;
    }

    private static Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                i -> specialTokensList.get(i),
                                i -> baseTokens + i)
                        );

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
    }

    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_1 -> new Q4_1FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_K -> new Q4_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q5_K -> new Q5_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q6_K -> new Q6_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case BF16 -> new BF16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new F16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}

record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {
    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    public static final class Configuration {
        public final int dim; // transformer dimension
        public final int hiddenDim; // for ffn layers
        public final int numberOfLayers; // number of layers
        public final int numberOfHeads; // number of query heads
        public final int numberOfKeyValueHeads; // number of key/value heads (can be < query heads because of multiquery)
        public final int vocabularySize; // vocabulary size, usually 256 (byte-level)
        public final int contextLength; // max sequence length
        public final float rmsNormEps;
        public final float ropeTheta;
        public final int headSize;

        Configuration withContextLength(int newContextLength) {
            if (newContextLength < 0) {
                return this; // no change
            }
            return new Configuration(this.dim, this.hiddenDim, this.numberOfLayers, this.numberOfHeads, this.numberOfKeyValueHeads, this.vocabularySize, newContextLength, this.rmsNormEps, this.ropeTheta);
        }

        public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
            this.dim = dim;
            this.hiddenDim = hiddenDim;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = dim / numberOfHeads;
        }
    }

    public static final class Weights {
        // token embedding table
        public final FloatTensor token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
        // weights for matmuls
        public final FloatTensor[] wq; // (layer, n_heads * head_size)
        public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
        public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
        public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)
        public final FloatBuffer[] rms_ffn_weight; // (layer, dim)
        // weights for ffn
        public final FloatTensor[] w1; // (layer, hidden_dim, dim)
        public final FloatTensor[] w2; // (layer, dim, hidden_dim)
        public final FloatTensor[] w3; // (layer, hidden_dim, dim)
        // public final rmsnorm
        public final FloatBuffer rms_final_weight; // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
        public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
        // (optional) classifier weights for the logits, on the last layer
        public final FloatTensor wcls; // (vocab_size, dim)

        public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.rms_ffn_weight = rms_ffn_weight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.rms_final_weight = rms_final_weight;
            this.freq_cis_real = freq_cis_real;
            this.freq_cis_imag = freq_cis_imag;
            this.wcls = wcls;
        }
    }

    public static final class State {

        // current wave of activations
        public final int batchsize;
        public final FloatTensor[] x; // activation at current time stamp (dim,)
        public final FloatTensor[] xb; // same, but inside a residual branch (dim,)
        public final FloatTensor[] xb2; // an additional buffer just for convenience (dim,)
        public final FloatTensor[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public final FloatTensor[] q; // query (dim,)
        public final FloatTensor[] k; // key (dim,)
        public final FloatTensor[] v; // value (dim,)
        public final FloatTensor[] att; // buffer for scores/attention values (n_heads, seq_len)
        public final FloatTensor logits; // output logits

        // kv cache
        public final FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
        public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)
        
        /** last index in previous block */
        int idxPrevBlock;

        public int latestToken;

        State(Configuration config, int batchsize) {
            this.batchsize = batchsize;
            this.x = allocate(batchsize, config.dim);
            this.xb = allocate(batchsize, config.dim);
            this.xb2 = allocate(batchsize, config.dim);
            this.hb = allocate(batchsize, config.hiddenDim);
            this.hb2 = allocate(batchsize, config.hiddenDim);
            this.q = allocate(batchsize, config.dim);
            this.k = allocate(batchsize, config.dim);
            this.v = allocate(batchsize, config.dim);
            this.att = allocate(batchsize, config.numberOfHeads, config.contextLength);
            idxPrevBlock = -1;

            this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
            int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
            this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
            this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        }
    }

    static FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    static FloatTensor forward(Llama model, State state, int[] tokens, int position, boolean computeLogits) {
        // a few convenience variables
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);
        final int nTokens = tokens.length;

        // copy the token embedding into x
        Parallel.parallelFor(0, nTokens, t ->
            weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim)
        );

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            // rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);
            final int curLayer = l;
            Parallel.parallelFor(0, nTokens, t ->
                rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps)
            );

            // qkv matmuls for this position
            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            Parallel.parallelFor(0, nTokens, t -> {
                for (int i = 0; i < dim; i += 2) {
                    int head_dim = i % headSize;
                    float fcr = weights.freq_cis_real.get((position + t) * (headSize / 2) + (head_dim / 2));
                    float fci = weights.freq_cis_imag.get((position + t) * (headSize / 2) + (head_dim / 2));
                    int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = vi == 0 ? state.q[t] : state.k[t]; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(i);
                        float v1 = vec.getFloat(i + 1);
                        vec.setFloat(i, v0 * fcr - v1 * fci);
                        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                    }
                }
            });

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });

            // If the logits are not required, the attention and FFN of the last layer can be skipped entirely.
            if (!computeLogits && curLayer == config.numberOfLayers - 1) {
                state.idxPrevBlock = nTokens - 1;
                return null;
            }

            // multihead attention. iterate over all heads
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int token = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position + token; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q[token].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att[token].setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att[token].softmaxInPlace(attOffset, position + token + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb[token].fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position + token; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att[token].getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(nTokens, state.xb, state.xb2, dim, dim);

            // residual connection back into x
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb2[t]);
            });

            // ffn rmsnorm
            Parallel.parallelFor(0, nTokens, t -> {
                rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps);
            });

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(nTokens, state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(nTokens, state.xb, state.hb2, config.hiddenDim, dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            });

            // elementwise multiply with w3(x)
            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].multiplyInPlace(state.hb2[t]);
            });

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(nTokens, state.hb, state.xb, dim, config.hiddenDim);

            // residual connection
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb[t]);
            });
        }

        // final rmsnorm
        Parallel.parallelFor(0, nTokens, t -> {
            rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps);
        });

        // classifier into logits
        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);
        state.idxPrevBlock = nTokens - 1;

        return state.logits;
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
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (promptIndex < promptTokens.size()) {
                final int nTokens = Math.min(maxTokens - position, Math.min(promptTokens.size() - promptIndex, state.batchsize));
                final int[] tokens = new int[nTokens];
                for (int i = 0; i < nTokens; i++) {
                    tokens[i] = promptTokens.get(promptIndex + i);
                    if (echo) {
                        // log prompt token (different color?)
                        System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(tokens[i]))));
                    }
                }
                if (echo) {
                    System.out.format("position=%d, promptIdx=%d, promptSize=%d, tokens=%s%n", position, promptIndex, promptTokens.size(), Arrays.toString(tokens));
                }
                // Only compute logits on the very last batch.
                boolean computeLogits = promptIndex + nTokens >= promptTokens.size();
                forward(model, state, tokens, position, computeLogits);
                position += nTokens - 1; // -1 -> incremented later in the for loop
                promptIndex += nTokens;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                startGen = System.nanoTime();
            } else {
                forward(model, state, new int[]{token}, position, true);
            }
            nextToken = sampler.sampleToken(state.logits);
            if (echo) {
                // log inferred token
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (stopTokens.contains(nextToken)) {
                break;
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        System.err.printf("%ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());

        return generatedTokens;
    }
}

/**
 * Byte Pair Encoding tokenizer.
 * <p>
 * Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows along the
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
class Tokenizer {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }

    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
     * if none_raise, then an error is raised if any special token is encountered in text
     * this is the default tiktoken behavior right now as well
     * any other behavior is either annoying, or a major footgun.
     */
    List<Integer> encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encodeOrdinary(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        // we can use re.split for this. note that surrounding the pattern with ()
        // makes it into a capturing group, so the special tokens will be included
        String specialPattern = special
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        // now all the special characters are separated from the rest of the text
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                ids.add(getSpecialTokens().get(part));
            } else {
                // this is an ordinary sequence, encode it normally
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0..255
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            // find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            // subtle: if there are no more merges available, the key will
            // result in an inf for every single pair, and the min will be
            // just the first pair in the list, arbitrarily
            // we can detect this terminating case by a membership check
            if (!this.merges.containsKey(pair)) {
                break; // nothing else can be merged anymore
            }
            // otherwise let's merge the best pair (lowest merge index)
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            // if not at the very last position AND the pair matches, replace it
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings.
     * The reversible bpe codes work on unicode strings.
     * This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
     * When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
     * This is a significant percentage of your normal, say, 32K bpe vocab.
     * To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
     * And avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        // return dict(zip(bs, cs))
        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    public static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encode(text)).boxed().toList();
    }

    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decoded.length(); i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }
}

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public static void parallelForLong(long startInclusive, long endExclusive, LongConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}

record Pair<First, Second>(First first, Second second) {
}

record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                       MemorySegment memorySegment) {
}

enum GGMLType {
    F32(Float.BYTES),
    F16(GGMLType.FLOAT16_BYTES),
    Q4_0(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_1(2 * GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // support has been removed
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // support has been removed
    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(GGMLType.FLOAT16_BYTES + 32 * Byte.BYTES, 32),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
    // k-quantizations
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q5_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + GGMLType.FLOAT16_BYTES, GGMLType.QK_K),
    Q8_K(Integer.MAX_VALUE),

    IQ2_XXS(Integer.MAX_VALUE),
    IQ2_XS(Integer.MAX_VALUE),
    IQ3_XXS(Integer.MAX_VALUE),
    IQ1_S(Integer.MAX_VALUE),
    IQ4_NL(Integer.MAX_VALUE),
    IQ3_S(Integer.MAX_VALUE),
    IQ2_S(Integer.MAX_VALUE),
    IQ4_XS(Integer.MAX_VALUE),

    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES),
    I64(Long.BYTES),
    F64(Double.BYTES),
    IQ1_M(Integer.MAX_VALUE),
    BF16(GGMLType.BFLOAT16_BYTES),
    Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    TQ1_0(Integer.MAX_VALUE),
    TQ2_0(Integer.MAX_VALUE);

    public static final int BFLOAT16_BYTES = 2;
    public static final int FLOAT16_BYTES = 2;

    private static final GGMLType[] VALUES = values();

    private final int typeSize;

    private final int blockSize;

    public int getTypeSize() {
        return typeSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public static GGMLType fromId(int id) {
        return VALUES[id];
    }

    GGMLType(int typeSize) {
        this(typeSize, 1);
    }

    public long byteSizeFor(long numberOfElements) {
        long t = numberOfElements * (long) getTypeSize();
        assert t % getBlockSize() == 0;
        return t / getBlockSize();
    }

    public static final int QK_K = 256; // or 64?

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

/**
 * Over-simplified, shapeless, float tensor.
 * <p>
 * Not a strict tensor, but rather just a sequence of floats, not required to be backed by memory
 * e.g. can represent a sequence of quantized floats.
 */
abstract class FloatTensor {
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    // static final ValueLayout.OfFloat JAVA_FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
    // static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);

    // The use of Unsafe in this file is a temporary workaround to support native-image.
    static final Unsafe UNSAFE;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    static short readShort(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        return UNSAFE.getShort(memorySegment.address() + offset);
    }

    static byte readByte(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        return UNSAFE.getByte(memorySegment.address() + offset);
    }

    static float readFloat(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead.
        return UNSAFE.getFloat(memorySegment.address() + offset);
    }

    static float readFloat16(MemorySegment memorySegment, long offset) {
        return Float.float16ToFloat(readShort(memorySegment, offset));
    }

    // Non-intrinsic FP16 -> FP32 conversion.
    // Kept local to avoid Graal Native Image issues observed with Float.float16ToFloat intrinsic.
    static float float16ToFloat(short h) {
        int bits = Short.toUnsignedInt(h);
        int sign = bits >>> 15;
        int exp = (bits >>> 10) & 0x1F;
        int frac = bits & 0x03FF;

        int outExp;
        int outFrac;

        if (exp == 0) {
            if (frac == 0) {
                // Zero.
                outExp = 0;
                outFrac = 0;
            } else {
                // Subnormal: normalize mantissa and adjust exponent.
                int e = -14;
                while ((frac & 0x0400) == 0) {
                    frac <<= 1;
                    e--;
                }
                frac &= 0x03FF;
                outExp = e + 127;
                outFrac = frac << 13;
            }
        } else if (exp == 0x1F) {
            // Inf / NaN.
            outExp = 0xFF;
            outFrac = frac << 13;
        } else {
            // Normalized number.
            outExp = exp - 15 + 127;
            outFrac = frac << 13;
        }

        int outBits = (sign << 31) | (outExp << 23) | outFrac;
        return Float.intBitsToFloat(outBits);
    }

    // Preferred vector size for the fast multiplication routines.
    // (Apple Silicon) NEON only supports up-to 128bit vectors.
    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    abstract int size();

    abstract float getFloat(long index);

    abstract void setFloat(int index, float value);

    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);

    abstract GGMLType type();

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

    public static long numberOfElementsLong(int... dimensions) {
        long result = 1;
        for (int d : dimensions) {
            assert d > 0;
            result = Math.multiplyExact(result, d);
        }
        return result;
    }

    static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }

    void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
        if (that.length != out.length) {
            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
        }
        Parallel.parallelForLong(0, dim0 * context, ti -> {
            int idxArr = (int) (ti / dim0);
            int i = (int) (ti % dim0);
            out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1)); 
        });
    }

    @FunctionalInterface
    interface AggregateFunction {
        float apply(float acc, float value);
    }

    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    int argmax() {
        return argmax(0, size());
    }

    @FunctionalInterface
    interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }

    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, size(), mapFunction);
    }

    FloatTensor mapWithIndexInPlace(int thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }

    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, size());
    }

    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor multiplyInPlace(FloatTensor that) {
        return multiplyInPlace(0, that, 0, size());
    }

    FloatTensor divideInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

    FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    FloatTensor softmaxInPlace(int thisOffset, int size) {
        // find max value (for numerical stability)
        float maxVal = max(thisOffset, size);
        // exp and sum
        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        // normalize
        return divideInPlace(thisOffset, size, sum);
    }

    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        // this[thatOffset ... thatOffset + size) = a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}

/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_0} format.
 * <p>
 * This tensor implementation is not compatible with {@link FloatTensor}, but
 * {@link #dot(int, FloatTensor, int, int)} has a vectorized implementation that is used when
 * the second argument implements {@link FloatTensor}.
 */
final class Q4_0FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q4_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_0.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = readFloat16(memorySegment, blockOffset);
        byte quant;
        int modIndex = (int) (index % GGMLType.Q4_0.getBlockSize());
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q4_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize();
        int upperBound = size / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_0.getBlockSize(), blockOffset += GGMLType.Q4_0.getTypeSize()) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = sum0.add(sum2).fma(wScale, val);
                }
                case 256 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 0) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q4_1FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q4_1FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override int size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q4_1; }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_1.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q4_1.getTypeSize();
        float delta = readFloat16(memorySegment, blockOffset);
        float min = readFloat16(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
        int modIndex = (int) (index % GGMLType.Q4_1.getBlockSize());
        int quant;
        if (modIndex < 16) {
            quant = Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * GGMLType.FLOAT16_BYTES + modIndex)) & 0x0F;
        } else {
            quant = (Byte.toUnsignedInt(readByte(memorySegment, blockOffset + 2L * GGMLType.FLOAT16_BYTES + modIndex - 16)) >>> 4) & 0x0F;
        }
        return delta * quant + min;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_1FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q4_1.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_1.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_1.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q4_1.getBlockSize() * GGMLType.Q4_1.getTypeSize();
        int upperBound = size / GGMLType.Q4_1.getBlockSize() * GGMLType.Q4_1.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_1.getBlockSize(), blockOffset += GGMLType.Q4_1.getTypeSize()) {
            float deltaValue = readFloat16(thiz.memorySegment, blockOffset);
            float minValue = readFloat16(thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
            var wDelta = FloatVector.broadcast(F_SPECIES, deltaValue);
            var wMin = FloatVector.broadcast(F_SPECIES, minValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + 2L * GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that1.mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).fma(wMin, val);
                }
                case 256 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var that2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length());
                    var that3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that2.mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that1.fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that3.fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).add(that2).add(that3).fma(wMin, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        s0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wDelta, val);
                    }
                    // Vectorized min contribution.
                    var thatSum = FloatVector.zero(F_SPECIES);
                    for (int k = 0; k < GGMLType.Q4_1.getBlockSize(); k += F_SPECIES.length()) {
                        thatSum = thatSum.add(that.getFloatVector(F_SPECIES, thatOffset + j + k));
                    }
                    val = thatSum.fma(wMin, val);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q4_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q4_K.getTypeSize();

    final int size;
    final MemorySegment memorySegment;

    public Q4_KFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override int size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q4_K; }

    // Decode 6-bit scale/min values from the packed 12-byte K-quant scale table.
    // ggml layout stores eight values with mixed low/high-bit packing.
    static int getScaleMinK4(int j, MemorySegment mem, long scalesOffset, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(mem, scalesOffset + idx)) & 63;
        }
        int lowIdx = j + 4;
        int highIdx = isMin ? j : j - 4;
        int low = isMin
                ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
                : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
        int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
        return low | (high << 4);
    }

    @Override
    public float getFloat(long index) {
        // Q4_K block layout (256 values):
        // - d (fp16), dmin (fp16)
        // - scales[12] packed (8 scales + 8 mins, each 6-bit)
        // - qs[128] packed nibbles (two 4-bit quants per byte)
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
        long scalesOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES;
        long qsOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12;

        // Each group of 64 values uses two sub-blocks:
        // - low nibble half (32 values)
        // - high nibble half (32 values)
        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        int subBlock;
        int nibbleIndex;
        boolean isHigh;
        if (inGroup < 32) {
            subBlock = group * 2;
            nibbleIndex = inGroup;
            isHigh = false;
        } else {
            subBlock = group * 2 + 1;
            nibbleIndex = inGroup - 32;
            isHigh = true;
        }

        int sc = getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32L + nibbleIndex);
        int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q4_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // Keep two accumulators to reduce dependency chain pressure.
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = size / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
            long scalesOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES;
            long qsOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12;

            // 4 groups * 64 quantized values = 256 values per block.
            for (int g = 0; g < 4; g++) {
                float d1 = d * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, false);
                float negM1 = -(dmin * getScaleMinK4(g * 2, thiz.memorySegment, scalesOff, true));
                float d2 = d * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, false);
                float negM2 = -(dmin * getScaleMinK4(g * 2 + 1, thiz.memorySegment, scalesOff, true));

                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);

                int loBase = thatOffset + j + g * 64;
                int hiBase = thatOffset + j + g * 64 + 32;

                // 32 bytes of packed nibbles per group, processed as two 16-byte chunks.
                for (int c = 0; c < 2; c++) {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qsOff + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

                    int loIdx = loBase + c * 16;
                    int hiIdx = hiBase + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQ0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            val2 = loQ1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + F_SPECIES.length()), val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = hiQ0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx), val);
                            val2 = hiQ1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQ.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loIdx + p * F_SPECIES.length()), val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = hiQ.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiIdx + p * F_SPECIES.length()), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q5_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q5_K.getTypeSize();

    final int size;
    final MemorySegment memorySegment;

    public Q5_KFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override int size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q5_K; }

    @Override
    public float getFloat(long index) {
        // Q5_K extends Q4_K by adding one high bit per quant in qh[32].
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
        long scalesOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES;
        long qhOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12;
        long qsOffset = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12 + 32;

        // Same 4 x 64 grouping strategy as Q4_K.
        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;

        int sc = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32L + l);
        int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);

        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(memorySegment, qhOffset + l)) >> qhBitPos) & 1;

        int quant = nibble | (qhBit << 4);
        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q5_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // Dual accumulators improve instruction-level parallelism.
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES);
            long scalesOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES;
            long qhOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12;
            long qsOff = blockOffset + 2L * GGMLType.FLOAT16_BYTES + 12 + 32;
            var qh0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff, ByteOrder.LITTLE_ENDIAN);
            var qh1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, qhOff + 16, ByteOrder.LITTLE_ENDIAN);

            for (int g = 0; g < 4; g++) {
                int loSubBlock = g * 2;
                int hiSubBlock = loSubBlock + 1;
                float d1 = d * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, false);
                float m1 = dmin * Q4_KFloatTensor.getScaleMinK4(loSubBlock, thiz.memorySegment, scalesOff, true);
                float d2 = d * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, false);
                float m2 = dmin * Q4_KFloatTensor.getScaleMinK4(hiSubBlock, thiz.memorySegment, scalesOff, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);

                for (int c = 0; c < 2; c++) {
                    int loBase = thatOffset + j + g * 64 + c * 16;
                    int hiBase = thatOffset + j + g * 64 + 32 + c * 16;

                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var loQ = wBytes.and((byte) 0xF);
                    var hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);

                    // Merge the 5th quantization bit from qh into low/high nibble lanes.
                    var qhBytes = c == 0 ? qh0 : qh1;
                    loQ = loQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hiQ = hiQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                        }
                        case 256 -> {
                            var loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            var hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQf0.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val = loQf1.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + F_SPECIES.length()), val);
                            val2 = hiQf0.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                            val2 = hiQf1.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQf.fma(d1Vec, negM1Vec).fma(that.getFloatVector(F_SPECIES, loBase + off), val);
                                val2 = hiQf.fma(d2Vec, negM2Vec).fma(that.getFloatVector(F_SPECIES, hiBase + off), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q6_KFloatTensor extends FloatTensor {

    static final int BLOCK_SIZE = GGMLType.QK_K;
    static final int TYPE_SIZE = GGMLType.Q6_K.getTypeSize();

    final int size;
    final MemorySegment memorySegment;

    public Q6_KFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override int size() { return size; }
    @Override public void setFloat(int index, float value) { throw new UnsupportedOperationException("setFloat"); }
    @Override FloatVector getFloatVector(VectorSpecies<Float> species, int index) { throw new UnsupportedOperationException("getFloatVector"); }
    @Override public GGMLType type() { return GGMLType.Q6_K; }

    @Override
    public float getFloat(long index) {
        // Q6_K block layout (256 values):
        // - ql[128]: low 4 bits for two halves
        // - qh[64]: two extra bits per quant
        // - scales[16]: signed int8 per 16-value stripe
        // - d (fp16): shared block scale
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128;
        long scOff = blockOffset + 192;
        float d = readFloat16(memorySegment, blockOffset + 192 + 16);

        // Two 128-value halves per block.
        int half = withinBlock / 128;
        int rem128 = withinBlock % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;

        long qlBase = qlOff + half * 64L;
        long qhBase = qhOff + half * 32L;

        int qlNibble;
        int qhShift;
        switch (sub32) {
            case 0 -> {
                qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) & 0xF;
                qhShift = 0;
            }
            case 1 -> {
                qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) & 0xF;
                qhShift = 2;
            }
            case 2 -> {
                qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) >> 4) & 0xF;
                qhShift = 4;
            }
            case 3 -> {
                qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) >> 4) & 0xF;
                qhShift = 6;
            }
            default -> throw new IllegalStateException();
        }

        int qhBits = (Byte.toUnsignedInt(readByte(memorySegment, qhBase + l)) >> qhShift) & 3;
        int q6 = (qlNibble | (qhBits << 4)) - 32;
        int sc = readByte(memorySegment, scOff + half * 8L + sub32 * 2L + l / 16); // signed int8

        return d * sc * q6;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDot(Q6_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // Separate accumulators for alternating stripes.
        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            // Avoid Float.float16ToFloat intrinsic here: Graal Native Image can miscompile this
            // specific hot loop. Use the local non-intrinsic conversion helper instead.
            float d = float16ToFloat(readShort(thiz.memorySegment, blockOffset + 192 + 16));

            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64L;
                long qhBase = qhOff + h * 32L;

                int base = thatOffset + j + h * 128;
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                            qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);

                    // Reconstruct signed 6-bit quants from ql nibbles + qh two-bit planes.
                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);

                    float ds0 = d * readByte(thiz.memorySegment, scOff + h * 8L + c);
                    float ds1 = d * readByte(thiz.memorySegment, scOff + h * 8L + 2 + c);
                    float ds2 = d * readByte(thiz.memorySegment, scOff + h * 8L + 4 + c);
                    float ds3 = d * readByte(thiz.memorySegment, scOff + h * 8L + 6 + c);

                    var ds0Vec = FloatVector.broadcast(F_SPECIES, ds0);
                    var ds1Vec = FloatVector.broadcast(F_SPECIES, ds1);
                    var ds2Vec = FloatVector.broadcast(F_SPECIES, ds2);
                    var ds3Vec = FloatVector.broadcast(F_SPECIES, ds3);

                    int sg0Idx = base + c * 16;
                    int sg1Idx = base + 32 + c * 16;
                    int sg2Idx = base + 64 + c * 16;
                    int sg3Idx = base + 96 + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx), val);
                            var q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx), val2);
                            var q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx), val);
                            var q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx), val2);
                        }
                        case 256 -> {
                            for (int p = 0; p < 2; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), val);
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), val2);
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), val);
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), val2);
                            }
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q0f.mul(ds0Vec).fma(that.getFloatVector(F_SPECIES, sg0Idx + off), val);
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q1f.mul(ds1Vec).fma(that.getFloatVector(F_SPECIES, sg1Idx + off), val2);
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = q2f.mul(ds2Vec).fma(that.getFloatVector(F_SPECIES, sg2Idx + off), val);
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = q3f.mul(ds3Vec).fma(that.getFloatVector(F_SPECIES, sg3Idx + off), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q8_0FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q8_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q8_0.getBlockSize();
        long withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
        long blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        byte quant = readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + withinBlockIndex);
        float scale = readFloat16(memorySegment, blockOffset);
        return quant * scale;
    }

    public static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + startIndex to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize();
        int upperBound = size / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q8_0.getBlockSize(), blockOffset += GGMLType.Q8_0.getTypeSize()) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).fma(wScale, val);
                }
                case 256 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class BF16FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public BF16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.BF16;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        return bfloat16ToFloat(readShort(memorySegment, index * GGMLType.BFLOAT16_BYTES));
    }

    private float bfloat16ToFloat(short bfloat16) {
        return Float.intBitsToFloat(bfloat16 << 16);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(BF16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * (long) GGMLType.BFLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            // BFloat16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───····───[7]─┬─[6]────····────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (7 bits)  │ BFloat16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            //    │             │                    │
            //    ▼             ▼                    ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            FloatVector thizVector = bfloat16
                    .castShape(I_SPECIES, 0) // (int) vi
                    .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                    .reinterpretAsFloats(); // Float.intBitsToFloat(vi)
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }

        return result;
    }
}

final class F16FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public F16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        return readFloat16(memorySegment, index * GGMLType.FLOAT16_BYTES);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(F16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bits16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * (long) GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);

            var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts(); // (int) bits16
            // Does not support infinities nor NaNs, preserves sign, emulate DAZ (denormals-are-zero).
            // Expects well-formed float16 values only (e.g. model weights).
            // Fast Float16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───···───[10]─┬─[9]────····────[0]─┐
            // │ Sign │ Exponent (5 bits) │ Mantissa (10 bits) │ Float16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            //    │             │                    │
            //    ▼             ▼                    ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            //
            // Shifts and adjustments:
            // - Sign:       float16[15] -> float32[31] (shift 16 bits up)
            // - Exponent:   float16[10-14] -> float32[23-30] (+ bias adjustment)
            // - Mantissa:   float16[0-9] -> float32[13-22] (shift 13 bits up)
            //
            // exp = bits32 & 0x7C00
            // zeroExponentMask = exp == 0 ? 0 : ~0
            var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31); // = (-exp) >> 31
            bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16) // sign
                    .or(
                            // exponent and mantissa combined
                            bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13)
                                    .and(zeroExponentMask) // -0, +0 and DAZ (denormals-are-zero)

                    );

            FloatVector thizVector = bits32.reinterpretAsFloats(); // Float.intBitsToFloat(vi)
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }

        return result;
    }
}

final class F32FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public F32FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromMemorySegment(species, memorySegment, index * (long) Float.BYTES, ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        return readFloat(memorySegment, index * (long) Float.BYTES);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(F32FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            FloatVector thizVector = FloatVector.fromMemorySegment(F_SPECIES, thiz.memorySegment, (thisOffset + i) * (long) Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }
        return result;
    }
}

final class ArrayFloatTensor extends FloatTensor {

    final float[] values;

    ArrayFloatTensor(float[] values) {
        this.values = values;
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float getFloat(long index) {
        return values[Math.toIntExact(index)];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }
}

final class RoPE {
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta,
                                                            boolean ropeScaling, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (ropeScaling) {
                    // Llama 3.1 scaling
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
}

record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
        return tokens.length;
    }
}

@FunctionalInterface
interface Sampler {
    int sampleToken(FloatTensor logits);

    Sampler ARGMAX = FloatTensor::argmax;
}

record CategoricalSampler(RandomGenerator rng) implements Sampler {

    @Override
    public int sampleToken(FloatTensor logits) {
        // sample index from probabilities (they must sum to 1!)
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return logits.size() - 1; // in case of rounding errors
    }
}

final class ToppSampler implements Sampler {

    final int[] indices;
    final float topp;
    final RandomGenerator rng;

    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    @Override
    public int sampleToken(FloatTensor logits) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        Comparator<Integer> comparator = Comparator.<Integer>comparingDouble(i -> logits.getFloat(i.longValue())).reversed();

        int n = logits.size();
        int head = 0;
        int tail = n - 1;
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.getFloat(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        // build heap O(n0)
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        // truncate the list where cumulative probability of the largest k elements exceeds topp
        // O(k lg n0)
        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += logits.getFloat(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break; // we've exceeded topp by including lastIndex
            }
            siftDown(indices, 0, i - 1, comparator);
        }

        // sample from the truncated list
        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.getFloat(indices[i]);
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex]; // in case of rounding errors
    }
}

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
class ChatFormat {

    final Tokenizer tokenizer;
    final int beginOfText;
    final int endHeader;
    final int startHeader;
    final int endOfTurn;
    final int endOfText;
    final int endOfMessage;
    final Set<Integer> stopTokens;

    public ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }

    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    public List<Integer> encodeHeader(ChatFormat.Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeMessage(ChatFormat.Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
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

/**
 * Support for AOT preloading of GGUF metadata with GraalVM's Native Image.
 *
 * <p>
 * To preload a model at build time, pass {@code -Dllama.PreloadGGUF=/path/to/model.gguf}
 * to the native-image builder command. At runtime, the preloaded model will be used
 * iff the specified and preloaded file names (base name) match.
 */
final class AOT {
    record PartialModel(String modelFileName, Llama model, long tensorDataOffset, Map<String, GGUF.GGUFTensorInfo> tensorInfos) {}

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("llama.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = GGUF.loadModel(fileChannel, path.toString());
                return new PartialModel(
                        path.getFileName().toString(),
                        ModelLoader.loadModel(fileChannel, gguf, Llama3.Options.DEFAULT_MAX_TOKENS, false),
                        gguf.getTensorDataOffset(),
                        gguf.getTensorInfos()
                );
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Tries to reuse a compatible AOT preloaded model.
     * The file name (base name) must match with the preloaded file name.
     * No checksum/hash is checked for performance reasons.
     */
    public static Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;
        if (preLoaded == null) {
            return null; // no pre-loaded model stored
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            // Preloaded and specified model file names didn't match.
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
             var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            // Load only the tensors (mmap slices).
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensorInfos());
            Llama.Weights weights = ModelLoader.loadWeights(tensorEntries, baseModel.configuration());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights);
        }
    }
}
