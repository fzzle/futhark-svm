from cffi import FFI
from os import path
import io

ffi = FFI()

here = path.abspath(path.dirname(__file__))

def strip_includes(f):
  return '\n'.join(line for line in f if not line[0] == '#')

with io.open(path.join(here, 'bin/fsvm.c')) as f:
  ffi.set_source(
    '_fsvm',
    f.read(),
    libraries=['OpenCL'],
    extra_compile_args=[
      '-O3', '-lm', '-std=c99', '-w'
    ]
  )

with io.open(path.join(here, 'bin/fsvm.h')) as f:
  ffi.cdef(
    "typedef void* cl_command_queue;\n"
    "typedef void* cl_mem;\n"
    f"{strip_includes(f)}\n"
    "void free(void *ptr);"
  )

if __name__ == '__main__':
  ffi.compile(verbose=True)