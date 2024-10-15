ifdef JAVA_HOME
	JAVAC ?= ${JAVA_HOME}/bin/javac
	JAVA ?= ${JAVA_HOME}/bin/java
	JAR ?= ${JAVA_HOME}/bin/jar
	NATIVE_IMAGE ?= ${JAVA_HOME}/bin/native-image
endif

JAVAC ?= javac
JAVA ?= java
JAR ?= jar
NATIVE_IMAGE ?= native-image

JAVA_MAJOR_VERSION := $(shell $(JAVA) -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)

JAVA_COMPILE_OPTIONS = --enable-preview -source $(JAVA_MAJOR_VERSION) -g --add-modules jdk.incubator.vector
JAVA_RUNTIME_OPTIONS = --enable-preview --add-modules jdk.incubator.vector

ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif

# Define the executable name
NATIVE_FILE := llama3$(EXE)

JAVA_MAIN_CLASS = com.llama4j.Llama3
JAR_FILE = llama3.jar

JAVA_SOURCES = $(wildcard *.java)
JAVA_CLASSES = $(patsubst %.java, target/classes/com/llama4j/%.class, $(JAVA_SOURCES))

# Bundle all classes in a jar
$(JAR_FILE): $(JAVA_CLASSES) LICENSE
	$(JAR) -cvfe $(JAR_FILE) $(JAVA_MAIN_CLASS) LICENSE -C target/classes .

jar: $(JAR_FILE)

# Compile the Java source files
compile: $(JAVA_CLASSES)

# Prints the command to run the Java main class
run-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -cp target/classes $(JAVA_MAIN_CLASS)

# Prints the command to run the $(JAR_FILE)
run-jar-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE)

# Clean the target directory
clean:
	rm -rf ./target
	rm $(JAR_FILE) $(NATIVE_FILE)

# Compile the Java source files
target/classes/com/llama4j/%.class: %.java
	$(JAVAC) $(JAVA_COMPILE_OPTIONS) -d target/classes $<

# Create the target directory
target/classes:
	mkdir -p target/classes

$(NATIVE_FILE): jar
	$(NATIVE_IMAGE) \
	 	-H:+UnlockExperimentalVMOptions \
		-H:+VectorAPISupport \
		-H:+ForeignAPISupport \
		-O3 \
		-march=native \
		--enable-preview \
		--add-modules jdk.incubator.vector \
		--initialize-at-build-time='com.llama4j.AOT,com.llama4j.FloatTensor,com.llama4j.' \
		-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
		-Dllama.PreloadGGUF=$(PRELOAD_GGUF) \
		-jar $(JAR_FILE) \
		-o $(NATIVE_FILE)

# Make the target directory a dependency of the Java class files
$(JAVA_CLASSES): target/classes
compile: target/classes
default: jar
native: $(NATIVE_FILE)

.PHONY: compile clean jar run-command run-jar-command
.SUFFIXES: .java .class .jar
