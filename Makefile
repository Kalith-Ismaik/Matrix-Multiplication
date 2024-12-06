# Compilers
NVCC      = nvcc         # NVIDIA CUDA compiler
CXX       = nvcc          # Standard C++ compiler

# Compiler Flags
CUFLAGS    = -arch=sm_80 -gencode=arch=compute_80,code=sm_80            # CUDA architecture and optimization flags
CXXFLAGS   = -std=c++11 -O3             # Standard C++11 and optimization flags
NVFLAGS    = -x cu
DEBUGFLAGS = -g                         # Debugging flags
LDFLAGS    = -lcudart -lcublas          # CUDA libraries to link against

# Source files
CXXSRCS   = MatOper.cpp Main.cpp GEMM.cpp
CUSRC     = MatMul.cu
OBJS      = MatOper.o Main.o GEMM.o MatMul.o

TARGET    = MatMul.x

# Default target
all: $(TARGET)

# Rule to compile CUDA source files
%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

# Rule to compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile C++ files that require CUDA runtime (for GEMM.cpp)
GEMM.o: GEMM.cpp
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) $(CUFLAGS) -c $< -o $@

# Rule to link the final executable
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(LDFLAGS)

# Clean the build
.PHONY: clean all
clean:
	rm -f *.o $(TARGET)

# Debug build
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: CUFLAGS += $(DEBUGFLAGS)
debug: $(TARGET)