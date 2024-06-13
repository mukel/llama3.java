ifdef JAVA_HOME
	JAVAC ?= ${JAVA_HOME}/bin/javac
	JAVA ?= ${JAVA_HOME}/bin/java
	JAR ?= ${JAVA_HOME}/bin/jar
endif

JAVAC ?= javac
JAVA ?= java
JAR ?= jar

JAVA_COMPILE_OPTIONS = --enable-preview -source 21 -g --add-modules jdk.incubator.vector
JAVA_RUNTIME_OPTIONS = --enable-preview --add-modules jdk.incubator.vector

JAVA_MAIN_CLASS = Llama3
JAR_FILE = llama3.jar

JAVA_SOURCES = $(wildcard *.java)
JAVA_CLASSES = $(patsubst %.java, target/classes/%.class, $(JAVA_SOURCES))

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
	rm $(JAR_FILE)

# Compile the Java source files
target/classes/%.class: %.java
	$(JAVAC) $(JAVA_COMPILE_OPTIONS) -d target/classes $<

# Create the target directory
target/classes:
	mkdir -p target/classes

# Make the target directory a dependency of the Java class files
$(JAVA_CLASSES): target/classes
compile: target/classes
default: jar

.PHONY: compile clean jar run-command run-jar-command
.SUFFIXES: .java .class .jar
