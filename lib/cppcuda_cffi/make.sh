#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/bin/
PATH=$CUDA_PATH:$PATH

TORCHLIBPATH=$(python get_lib_path.py 2>&1)
echo $TORCHLIBPATH

cd src/cpp/

echo "Compiling roi_align_cpu.cpp with g++..."
g++ -I $TORCHLIBPATH -o roi_align_cpu_loop.o roi_align_cpu_loop.cpp -fPIC -shared -std=c++0x

echo "Compiling roi_align_forward_cuda_kernel.cu with nvcc..."
cd ../cuda/
nvcc -c -o roi_align_forward_cuda_kernel.cu.o roi_align_forward_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
nvcc -c -o roi_align_backward_cuda_kernel.cu.o roi_align_backward_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../

python bind.py