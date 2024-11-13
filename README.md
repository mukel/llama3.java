# Llama3.java

Practical [Llama 3](https://github.com/meta-llama/llama3), [3.1](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1) and [3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) inference implemented in a single Java file.

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/69bbf681-ae84-4a46-bcd6-746dbd421a6e">
</p>

This project is the successor of [llama2.java](https://github.com/mukel/llama2.java)
based on [llama2.c](https://github.com/karpathy/llama2.c) by [Andrej Karpathy](https://twitter.com/karpathy) and his [excellent educational videos](https://www.youtube.com/c/AndrejKarpathy).

Besides the educational value, this project will be used to test and tune compiler optimizations and features on the JVM, particularly for the [Graal compiler](https://www.graalvm.org/latest/reference-manual/java/compiler).

## Features

 - Single file, no dependencies
 - [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) parser
 - Llama 3 tokenizer based on [minbpe](https://github.com/karpathy/minbpe)
 - Llama 3 inference with Grouped-Query Attention
 - Support Llama 3.1 (ad-hoc RoPE scaling) and 3.2 (tie word embeddings)
 - Support for Q8_0 and Q4_0 quantizations
 - Fast matrix-vector multiplication routines for quantized tensors using Java's [Vector API](https://openjdk.org/jeps/469)
 - Simple CLI with `--chat` and `--instruct` modes.
 - GraalVM's Native Image support (EA builds [here](https://github.com/graalvm/oracle-graalvm-ea-builds))
 - AOT model pre-loading for instant time-to-first-token

**Interactive `--chat` mode in action:**
<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/f609bb73-7f11-4ea0-9ec7-43fbd3c96d3b">
</p>

## [Practical LLM inference in modern Java](https://www.youtube.com/watch?v=zgAMxC7lzkc)
**Presented at Devoxx Belgium, 2024**
<div align="center">
  <a href="https://www.youtube.com/watch?v=zgAMxC7lzkc">
    <img src="https://img.youtube.com/vi/zgAMxC7lzkc/sddefault.jpg">
  </a>
</div>

## Setup

Download pure `Q4_0` and (optionally) `Q8_0` quantized .gguf files from:
  - https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF
  - https://huggingface.co/mukel/Llama-3.2-3B-Instruct-GGUF
  - https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF
  - https://huggingface.co/mukel/Meta-Llama-3-8B-Instruct-GGUF

The pure `Q4_0` quantized models are recommended, except for the very small models (1B), please be gentle with [huggingface.co](https://huggingface.co) servers: 
```
# Llama 3.2 (3B)
curl -L -O https://huggingface.co/mukel/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf

# Llama 3.2 (1B)
curl -L -O https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

# Llama 3.1 (8B)
curl -L -O https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf

# Llama 3 (8B)
curl -L -O https://huggingface.co/mukel/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf

# Optionally download the Q8_0 quantized models
# curl -L -O https://huggingface.co/mukel/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf
# curl -L -O https://huggingface.co/mukel/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
```

#### Optional: quantize to pure `Q4_0` manually

In the wild, `Q8_0` quantizations are fine, but `Q4_0` quantizations are rarely pure e.g. the `token_embd.weights`/`output.weights` tensor are quantized with `Q6_K`, instead of `Q4_0`.  
A **pure** `Q4_0` quantization can be generated from a high precision (F32, F16, BFLOAT16) .gguf source 
with the `llama-quantize` utility from [llama.cpp](https://github.com/ggerganov/llama.cpp) as follows:

```bash
./llama-quantize --pure ./Meta-Llama-3-8B-Instruct-F32.gguf ./Meta-Llama-3-8B-Instruct-Q4_0.gguf Q4_0
```

## Build and run

Java 21+ is required, in particular the [`MemorySegment` mmap-ing feature](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.Arena)).

[`jbang`](https://www.jbang.dev/) is a perfect fit for this use case, just:
```
jbang Llama3.java --help
```
Or execute directly, also via [`jbang`](https://www.jbang.dev/):
```bash 
chmod +x Llama3.java
./Llama3.java --help
```

## Run from source

```bash
java --enable-preview --source 21 --add-modules jdk.incubator.vector LLama3.java -i --model Meta-Llama-3-8B-Instruct-Q4_0.gguf
```

#### Optional: Makefile + manually build and run

A simple [Makefile](./Makefile) is provided, run `make` to produce `llama3.jar` or manually:
```bash
javac -g --enable-preview -source 21 --add-modules jdk.incubator.vector -d target/classes Llama3.java
jar -cvfe llama3.jar com.llama4j.Llama3 LICENSE -C target/classes .
```

Run the resulting `llama3.jar` as follows: 
```bash
java --enable-preview --add-modules jdk.incubator.vector -jar llama3.jar --help
```

### GraalVM Native Image

Compile to native via `make` (recommended):

```bash
make native
```
Or directly:

```bash
native-image -H:+UnlockExperimentalVMOptions -H:+VectorAPISupport -H:+ForeignAPISupport -O3 -march=native --enable-preview --add-modules jdk.incubator.vector --initialize-at-build-time=com.llama4j.FloatTensor -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 -jar llama3.jar -o llama3
```

Run as Native Image:

```bash
./llama3 --model Llama-3.2-1B-Instruct-Q8_0 --chat
```

### AOT model preloading

`Llama3.java` supports AOT model preloading, enabling **0-overhead, instant inference, with minimal TTFT (time-to-first-token)**.

To AOT pre-load a GGUF model:
```bash
PRELOAD_GGUF=/path/to/model.gguf make native
```

A specialized, larger binary will be generated, with no parsing overhead for that particular model.
It can still run other models, although incurring the usual parsing overhead.

## Performance

GraalVM now supports more [Vector API](https://openjdk.org/jeps/469) operations. To give it a try, you need GraalVM for JDK 24 – get the EA builds from [`oracle-graalvm-ea-builds`](https://github.com/graalvm/oracle-graalvm-ea-builds) or sdkman: `sdk install java 24.ea.20-graal`.  
By default, the "preferred" vector size is used, it can be force-set with `-Dllama.VectorBitSize=0|128|256|512`, `0` means disabled.

#### llama.cpp

Vanilla `llama.cpp` built with `make`.
```bash
./llama-cli --version                                                                                                                                                                          130 ↵
version: 3862 (3f1ae2e3)
built with cc (GCC) 14.2.1 20240805 for x86_64-pc-linux-gnu
```

Executed as follows:
```bash
./llama-bench -m Llama-3.2-1B-Instruct-Q4_0.gguf -p 0 -n 128
```

#### Llama3.java

```bash
taskset -c 0-15 ./llama3 \
  --model ./Llama-3-1B-Instruct-Q4_0.gguf \
  --max-tokens 128 \
  --seed 42 \
  --stream false \
  --prompt "Why is the sky blue?"
```

Hardware specs: 2019 AMD Ryzen 3950X 16C/32T 64GB (3800) Linux 6.6.47.

****Notes**  
*Running on a single CCD e.g. `taskset -c 0-15 ./llama3 ...` since inference is constrained by memory bandwidth.* 

### Results
<p align="center">
  <img src="https://github.com/user-attachments/assets/7f36f26a-6a78-46b7-9067-fcbe7717aa44">
</p>

## License

MIT
