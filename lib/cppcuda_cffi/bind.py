import os
import torch
from torch.utils.ffi import create_extension


sources = ['src/roi_align_forward_cpu.c']
headers = ['src/roi_align_forward_cpu.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_align_forward_cuda.c','src/roi_align_backward_cuda.c']
    headers += ['src/roi_align_forward_cuda.h','src/roi_align_backward_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cpp/roi_align_cpu_loop.o',
                 'src/cuda/roi_align_forward_cuda_kernel.cu.o',
                 'src/cuda/roi_align_backward_cuda_kernel.cu.o']

extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    'roialign',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=['-std=c11']
)

if __name__ == '__main__':
    ffi.build()
