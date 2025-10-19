# Makefile for CUDA Shock Tube Simulation

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_50

# Compiler flags
NVCC_FLAGS = -O3 -std=c++14 $(CUDA_ARCH) -Xcompiler -fPIC
CXX_FLAGS = -O3 -std=c++14 -fPIC

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories
LIB_DIRS = -L/usr/local/cuda/lib64

# Libraries
LIBS = -lcudart -lcuda

# Source files
SOURCES = main.cpp shock_tube.cu cuda_kernels.cu
OBJECTS = main.o shock_tube.o cuda_kernels.o

# Target executable
TARGET = shock_tube

# Results directory
RESULTS_DIR = results

# Default target
all: $(TARGET) $(RESULTS_DIR)

# Create results directory
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(LIB_DIRS) -o $@ $^ $(LIBS)

# Compile main.cpp
main.o: main.cpp shock_tube.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile shock_tube.cu
shock_tube.o: shock_tube.cu shock_tube.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile cuda_kernels.cu
cuda_kernels.o: cuda_kernels.cu shock_tube.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(RESULTS_DIR)/*
	rm -f $(OBJECTS) $(TARGET) *.dat

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit g++ make

# Run tests
test: $(TARGET) $(RESULTS_DIR)
	./$(TARGET) --test

# Run Lax-Wendroff simulation
run-lax: $(TARGET) $(RESULTS_DIR)
	./$(TARGET) -s lax -n 256 -t 0.2 -o $(RESULTS_DIR)/lax_solution

# Run Roe simulation
run-roe: $(TARGET) $(RESULTS_DIR)
	./$(TARGET) -s roe -n 256 -t 0.2 -o $(RESULTS_DIR)/roe_solution

# Run both schemes
run-all: $(TARGET) $(RESULTS_DIR)
	./$(TARGET) -s lax -n 256 -t 0.2 -o $(RESULTS_DIR)/comparison
	./$(TARGET) -s roe -n 256 -t 0.2 -o $(RESULTS_DIR)/comparison

# Run default simulation (for main.py)
run-default: $(TARGET) $(RESULTS_DIR)
	./$(TARGET)

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG
debug: $(TARGET)

# Profile build
profile: NVCC_FLAGS += -pg
profile: CXX_FLAGS += -pg
profile: $(TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Run unit tests"
	@echo "  run-lax      - Run Lax-Wendroff simulation"
	@echo "  run-roe      - Run Roe simulation"
	@echo "  run-all      - Run both simulations"
	@echo "  run-default  - Run default simulation"
	@echo "  debug        - Build with debug symbols"
	@echo "  profile      - Build with profiling support"
	@echo "  install-deps - Install required dependencies"
	@echo "  help         - Show this help message"

# Phony targets
.PHONY: all clean test run-lax run-roe run-all run-default debug profile install-deps help
