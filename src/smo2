#!/usr/bin/env python
import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, value) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and self.device.type == device_type:
               if type(value) == str:
                   sizes[size] = self.device.get_info(getattr(cl.device_info,value))
               else:
                   sizes[size] = value
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0
    self.max_local_memory = int(self.device.local_mem_size)
    self.free_list = {}

    self.global_failure = self.pool.allocate(np.int32().itemsize)
    cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)
    self.global_failure_args = self.pool.allocate(np.int32().itemsize *
                                                  (self.global_failure_args_max+1))
    self.failure_is_an_option = np.int32(0)

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            # Bespoke sizes have no limit or default.
            max_value = None
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()

def sync(self):
    failure = np.empty(1, dtype=np.int32)
    cl.enqueue_copy(self.queue, failure, self.global_failure, is_blocking=True)
    self.failure_is_an_option = np.int32(0)
    if failure[0] >= 0:
        failure_args = np.empty(self.global_failure_args_max+1, dtype=np.int32)
        cl.enqueue_copy(self.queue, failure_args, self.global_failure_args, is_blocking=True)
        raise Exception(self.failure_msgs[failure[0]].format(*failure_args))
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
}
__kernel void map_transpose_f64(__local volatile
                                int64_t *block_11_backing_aligned_0,
                                int32_t destoffset_1, int32_t srcoffset_3,
                                int32_t num_arrays_4, int32_t x_elems_5,
                                int32_t y_elems_6, int32_t in_elems_7,
                                int32_t out_elems_8, int32_t mulx_9,
                                int32_t muly_10, __global
                                unsigned char *destmem_0, __global
                                unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),
                                        y_elems_6);
    int32_t odata_offset_33 = add32(squot32(destoffset_1, 8),
                                    our_array_offset_30);
    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 8),
                                    our_array_offset_30);
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = add32(mul32(get_group_id_1_41, 32), get_local_id_1_39);
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = add32(mul32(add32(y_index_32, mul32(j_43, 8)),
                                              x_elems_5), x_index_31);
            
            if (slt32(add32(y_index_32, mul32(j_43, 8)), y_elems_6) &&
                slt32(index_in_35, in_elems_7)) {
                ((__local
                  double *) block_11)[add32(mul32(add32(get_local_id_1_39,
                                                        mul32(j_43, 8)), 33),
                                            get_local_id_0_38)] = ((__global
                                                                    double *) srcmem_2)[add32(idata_offset_34,
                                                                                              index_in_35)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = add32(mul32(get_group_id_1_41, 32), get_local_id_0_38);
    y_index_32 = add32(mul32(get_group_id_0_40, 32), get_local_id_1_39);
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = add32(mul32(add32(y_index_32, mul32(j_43,
                                                                       8)),
                                               y_elems_6), x_index_31);
            
            if (slt32(add32(y_index_32, mul32(j_43, 8)), x_elems_5) &&
                slt32(index_out_36, out_elems_8)) {
                ((__global double *) destmem_0)[add32(odata_offset_33,
                                                      index_out_36)] = ((__local
                                                                         double *) block_11)[add32(add32(mul32(get_local_id_0_38,
                                                                                                               33),
                                                                                                         get_local_id_1_39),
                                                                                                   mul32(j_43,
                                                                                                         8))];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f64_low_height(__local volatile
                                           int64_t *block_11_backing_aligned_0,
                                           int32_t destoffset_1,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8, int32_t mulx_9,
                                           int32_t muly_10, __global
                                           unsigned char *destmem_0, __global
                                           unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),
                                        y_elems_6);
    int32_t odata_offset_33 = add32(squot32(destoffset_1, 8),
                                    our_array_offset_30);
    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 8),
                                    our_array_offset_30);
    int32_t x_index_31 = add32(add32(mul32(mul32(get_group_id_0_40, 16),
                                           mulx_9), get_local_id_0_38),
                               mul32(srem32(get_local_id_1_39, mulx_9), 16));
    int32_t y_index_32 = add32(mul32(get_group_id_1_41, 16),
                               squot32(get_local_id_1_39, mulx_9));
    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        ((__local double *) block_11)[add32(mul32(get_local_id_1_39, 17),
                                            get_local_id_0_38)] = ((__global
                                                                    double *) srcmem_2)[add32(idata_offset_34,
                                                                                              index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = add32(mul32(get_group_id_1_41, 16), squot32(get_local_id_0_38,
                                                             mulx_9));
    y_index_32 = add32(add32(mul32(mul32(get_group_id_0_40, 16), mulx_9),
                             get_local_id_1_39), mul32(srem32(get_local_id_0_38,
                                                              mulx_9), 16));
    
    int32_t index_out_36 = add32(mul32(y_index_32, y_elems_6), x_index_31);
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        ((__global double *) destmem_0)[add32(odata_offset_33, index_out_36)] =
            ((__local double *) block_11)[add32(mul32(get_local_id_0_38, 17),
                                                get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f64_low_width(__local volatile
                                          int64_t *block_11_backing_aligned_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t in_elems_7,
                                          int32_t out_elems_8, int32_t mulx_9,
                                          int32_t muly_10, __global
                                          unsigned char *destmem_0, __global
                                          unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),
                                        y_elems_6);
    int32_t odata_offset_33 = add32(squot32(destoffset_1, 8),
                                    our_array_offset_30);
    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 8),
                                    our_array_offset_30);
    int32_t x_index_31 = add32(mul32(get_group_id_0_40, 16),
                               squot32(get_local_id_0_38, muly_10));
    int32_t y_index_32 = add32(add32(mul32(mul32(get_group_id_1_41, 16),
                                           muly_10), get_local_id_1_39),
                               mul32(srem32(get_local_id_0_38, muly_10), 16));
    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);
    
    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&
                                         slt32(index_in_35, in_elems_7))) {
        ((__local double *) block_11)[add32(mul32(get_local_id_1_39, 17),
                                            get_local_id_0_38)] = ((__global
                                                                    double *) srcmem_2)[add32(idata_offset_34,
                                                                                              index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = add32(add32(mul32(mul32(get_group_id_1_41, 16), muly_10),
                             get_local_id_0_38), mul32(srem32(get_local_id_1_39,
                                                              muly_10), 16));
    y_index_32 = add32(mul32(get_group_id_0_40, 16), squot32(get_local_id_1_39,
                                                             muly_10));
    
    int32_t index_out_36 = add32(mul32(y_index_32, y_elems_6), x_index_31);
    
    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&
                                         slt32(index_out_36, out_elems_8))) {
        ((__global double *) destmem_0)[add32(odata_offset_33, index_out_36)] =
            ((__local double *) block_11)[add32(mul32(get_local_id_0_38, 17),
                                                get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void map_transpose_f64_small(__local volatile
                                      int64_t *block_11_backing_aligned_0,
                                      int32_t destoffset_1, int32_t srcoffset_3,
                                      int32_t num_arrays_4, int32_t x_elems_5,
                                      int32_t y_elems_6, int32_t in_elems_7,
                                      int32_t out_elems_8, int32_t mulx_9,
                                      int32_t muly_10, __global
                                      unsigned char *destmem_0, __global
                                      unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_11_backing_0 = (__local volatile
                                                          char *) block_11_backing_aligned_0;
    __local char *block_11;
    
    block_11 = (__local char *) block_11_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = mul32(squot32(get_global_id_0_37,
                                                mul32(y_elems_6, x_elems_5)),
                                        mul32(y_elems_6, x_elems_5));
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, mul32(y_elems_6,
                                                                  x_elems_5)),
                                 y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = add32(squot32(destoffset_1, 8),
                                    our_array_offset_30);
    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 8),
                                    our_array_offset_30);
    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);
    int32_t index_out_36 = add32(mul32(x_index_31, y_elems_6), y_index_32);
    
    if (slt32(get_global_id_0_37, in_elems_7)) {
        ((__global double *) destmem_0)[add32(odata_offset_33, index_out_36)] =
            ((__global double *) srcmem_2)[add32(idata_offset_34, index_in_35)];
    }
    
  error_0:
    return;
}
__kernel void replicate_9437(__global unsigned char *mem_9433,
                             int32_t num_elems_9434, double val_9435)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_9437;
    int32_t replicate_ltid_9438;
    int32_t replicate_gid_9439;
    
    replicate_gtid_9437 = get_global_id(0);
    replicate_ltid_9438 = get_local_id(0);
    replicate_gid_9439 = get_group_id(0);
    if (slt32(replicate_gtid_9437, num_elems_9434)) {
        ((__global double *) mem_9433)[replicate_gtid_9437] = val_9435;
    }
    
  error_0:
    return;
}
__kernel void segmap_8956(__global int *global_failure, int32_t n_8716,
                          int32_t loopres_8751, int32_t loopres_8785,
                          double deltaAi_8908, double deltaAj_8909, __global
                          unsigned char *mem_9372, __global
                          unsigned char *G_mem_9380, __global
                          unsigned char *mem_9392)
{
    #define segmap_group_sizze_8960 (train_binaryzisegmap_group_sizze_8959)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_9452;
    int32_t local_tid_9453;
    int32_t group_sizze_9456;
    int32_t wave_sizze_9455;
    int32_t group_tid_9454;
    
    global_tid_9452 = get_global_id(0);
    local_tid_9453 = get_local_id(0);
    group_sizze_9456 = get_local_size(0);
    wave_sizze_9455 = LOCKSTEP_WIDTH;
    group_tid_9454 = get_group_id(0);
    
    int32_t phys_tid_8956 = global_tid_9452;
    int32_t gtid_8955 = add32(mul32(group_tid_9454, segmap_group_sizze_8960),
                              local_tid_9453);
    
    if (slt32(gtid_8955, n_8716)) {
        double x_8968 = ((__global double *) G_mem_9380)[gtid_8955];
        double x_8969 = ((__global double *) mem_9372)[add32(mul32(gtid_8955,
                                                                   n_8716),
                                                             loopres_8751)];
        double y_8970 = deltaAi_8908 * x_8969;
        double x_8971 = x_8968 + y_8970;
        double x_8972 = ((__global double *) mem_9372)[add32(mul32(gtid_8955,
                                                                   n_8716),
                                                             loopres_8785)];
        double y_8973 = deltaAj_8909 * x_8972;
        double res_8974 = x_8971 + y_8973;
        
        ((__global double *) mem_9392)[gtid_8955] = res_8974;
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_8960
}
__kernel void segmap_8976(__global int *global_failure, int32_t n_8716, __global
                          unsigned char *mem_9383, __global
                          unsigned char *mem_9386, __global
                          unsigned char *mem_9389)
{
    #define segmap_group_sizze_8980 (train_binaryzisegmap_group_sizze_8979)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_9457;
    int32_t local_tid_9458;
    int32_t group_sizze_9461;
    int32_t wave_sizze_9460;
    int32_t group_tid_9459;
    
    global_tid_9457 = get_global_id(0);
    local_tid_9458 = get_local_id(0);
    group_sizze_9461 = get_local_size(0);
    wave_sizze_9460 = LOCKSTEP_WIDTH;
    group_tid_9459 = get_group_id(0);
    
    int32_t phys_tid_8976 = global_tid_9457;
    int32_t write_i_8975 = add32(mul32(group_tid_9459, segmap_group_sizze_8980),
                                 local_tid_9458);
    
    if (slt32(write_i_8975, 2)) {
        int32_t write_index_8924 = ((__global
                                     int32_t *) mem_9383)[write_i_8975];
        double write_value_8925 = ((__global double *) mem_9386)[write_i_8975];
        
        if (sle32(0, write_index_8924) && slt32(write_index_8924, n_8716)) {
            ((__global double *) mem_9389)[write_index_8924] = write_value_8925;
        }
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_8980
}
__kernel void segmap_intragroup_9007(__global int *global_failure,
                                     __local volatile
                                     int64_t *mem_9358_backing_aligned_0,
                                     __local volatile
                                     int64_t *mem_9353_backing_aligned_1,
                                     __local volatile
                                     int64_t *mem_9343_backing_aligned_2,
                                     __local volatile
                                     int64_t *mem_9338_backing_aligned_3,
                                     int32_t n_8716, int32_t m_8717,
                                     int32_t num_groups_x_9002,
                                     int32_t num_whole_tiles_9008,
                                     int32_t residual_input_9137,
                                     unsigned char cond_9138, __global
                                     unsigned char *xs_mem_9324, __global
                                     unsigned char *ys_mem_9325, __global
                                     unsigned char *mem_9329, __global
                                     unsigned char *mem_9372)
{
    #define tile_sizze_8998 (train_binaryzitile_sizze_8997)
    #define group_sizze_8999 (mul32(train_binaryzitile_sizze_8997, train_binaryzitile_sizze_8997))
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict mem_9358_backing_3 = (__local volatile
                                                          char *) mem_9358_backing_aligned_0;
    __local volatile char *restrict mem_9353_backing_2 = (__local volatile
                                                          char *) mem_9353_backing_aligned_1;
    __local volatile char *restrict mem_9343_backing_1 = (__local volatile
                                                          char *) mem_9343_backing_aligned_2;
    __local volatile char *restrict mem_9338_backing_0 = (__local volatile
                                                          char *) mem_9338_backing_aligned_3;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_9420;
    int32_t local_tid_9421;
    int32_t group_sizze_9424;
    int32_t wave_sizze_9423;
    int32_t group_tid_9422;
    
    global_tid_9420 = get_global_id(0);
    local_tid_9421 = get_local_id(0);
    group_sizze_9424 = get_local_size(0);
    wave_sizze_9423 = LOCKSTEP_WIDTH;
    group_tid_9422 = get_group_id(0);
    
    int32_t gid_flat_9007 = group_tid_9422;
    int32_t gid_x_8995 = squot32(group_tid_9422, num_groups_x_9002);
    int32_t gid_y_8996;
    
    gid_y_8996 = sub32(group_tid_9422, mul32(squot32(group_tid_9422,
                                                     num_groups_x_9002),
                                             num_groups_x_9002));
    
    double mem_9333[1];
    int32_t ltid_x_9024 = squot32(local_tid_9421, tile_sizze_8998);
    int32_t ltid_y_9025;
    
    ltid_y_9025 = sub32(local_tid_9421, mul32(squot32(local_tid_9421,
                                                      tile_sizze_8998),
                                              tile_sizze_8998));
    
    int32_t ltid_flat_9026;
    
    ltid_flat_9026 = local_tid_9421;
    if (slt32(ltid_x_9024, tile_sizze_8998) && slt32(ltid_y_9025,
                                                     tile_sizze_8998)) {
        mem_9333[mul32(0, tile_sizze_8998)] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t binop_x_9101 = mul32(gid_x_8995, tile_sizze_8998);
    int32_t binop_x_9103 = mul32(gid_y_8996, tile_sizze_8998);
    __local char *mem_9338;
    
    mem_9338 = (__local char *) mem_9338_backing_0;
    
    __local char *mem_9343;
    
    mem_9343 = (__local char *) mem_9343_backing_1;
    
    double mem_9347[1];
    
    for (int32_t tile_id_9035 = 0; tile_id_9035 < num_whole_tiles_9008;
         tile_id_9035++) {
        int32_t binop_x_9097 = mul32(tile_sizze_8998, tile_id_9035);
        int32_t ltid_x_9036 = squot32(local_tid_9421, tile_sizze_8998);
        int32_t ltid_y_9037;
        
        ltid_y_9037 = sub32(local_tid_9421, mul32(squot32(local_tid_9421,
                                                          tile_sizze_8998),
                                                  tile_sizze_8998));
        
        int32_t ltid_flat_9038;
        
        ltid_flat_9038 = local_tid_9421;
        if (slt32(ltid_x_9036, tile_sizze_8998) && slt32(ltid_y_9037,
                                                         tile_sizze_8998)) {
            int32_t i_9098 = add32(ltid_x_9036, binop_x_9097);
            int32_t j_9100 = add32(ltid_y_9037, binop_x_9097);
            int32_t gtid_9102 = add32(ltid_x_9036, binop_x_9101);
            int32_t gtid_9104 = add32(ltid_y_9037, binop_x_9103);
            double tile_elem_9107 = ((__global
                                      double *) xs_mem_9324)[add32(mul32(gtid_9102,
                                                                         m_8717),
                                                                   j_9100)];
            double tile_elem_9108 = ((__global
                                      double *) mem_9329)[add32(mul32(i_9098,
                                                                      n_8716),
                                                                gtid_9104)];
            
            ((__local double *) mem_9338)[add32(mul32(ltid_x_9036,
                                                      tile_sizze_8998),
                                                ltid_y_9037)] = tile_elem_9107;
            ((__local double *) mem_9343)[add32(mul32(ltid_x_9036,
                                                      tile_sizze_8998),
                                                ltid_y_9037)] = tile_elem_9108;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_9061 = squot32(local_tid_9421, tile_sizze_8998);
        int32_t ltid_y_9062;
        
        ltid_y_9062 = sub32(local_tid_9421, mul32(squot32(local_tid_9421,
                                                          tile_sizze_8998),
                                                  tile_sizze_8998));
        
        int32_t ltid_flat_9063;
        
        ltid_flat_9063 = local_tid_9421;
        if (slt32(ltid_x_9061, tile_sizze_8998) && slt32(ltid_y_9062,
                                                         tile_sizze_8998)) {
            int32_t gtid_9111 = add32(ltid_x_9061, binop_x_9101);
            int32_t gtid_9113 = add32(ltid_y_9062, binop_x_9103);
            double acc_9116 = mem_9333[mul32(0, tile_sizze_8998)];
            bool binop_x_9119 = slt32(gtid_9111, n_8716);
            bool binop_y_9120 = slt32(gtid_9113, n_8716);
            bool cond_9121 = binop_x_9119 && binop_y_9120;
            double acc_9122;
            
            if (cond_9121) {
                double x_9123;
                double redout_9296 = acc_9116;
                
                for (int32_t i_9297 = 0; i_9297 < tile_sizze_8998; i_9297++) {
                    double x_9127 = ((__local
                                      double *) mem_9338)[add32(mul32(ltid_x_9061,
                                                                      tile_sizze_8998),
                                                                i_9297)];
                    double x_9128 = ((__local
                                      double *) mem_9343)[add32(mul32(i_9297,
                                                                      tile_sizze_8998),
                                                                ltid_y_9062)];
                    double res_9129 = x_9127 * x_9128;
                    double res_9126 = res_9129 + redout_9296;
                    double redout_tmp_9426 = res_9126;
                    
                    redout_9296 = redout_tmp_9426;
                }
                x_9123 = redout_9296;
                acc_9122 = x_9123;
            } else {
                acc_9122 = acc_9116;
            }
            mem_9347[mul32(0, tile_sizze_8998)] = acc_9122;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_9427 = 0; i_9427 <
             squot32(sub32(add32(sub32(mul32(tile_sizze_8998, tile_sizze_8998),
                                       local_tid_9421), group_sizze_8999), 1),
                     group_sizze_8999); i_9427++) {
            mem_9333[mul32(0, tile_sizze_8998)] = mem_9347[mul32(0,
                                                                 tile_sizze_8998)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    __local char *mem_9353;
    
    mem_9353 = (__local char *) mem_9353_backing_2;
    
    __local char *mem_9358;
    
    mem_9358 = (__local char *) mem_9358_backing_3;
    
    double mem_9362[1];
    double mem_9408[1];
    
    if (cond_9138) {
        for (int32_t i_9428 = 0; i_9428 <
             squot32(sub32(add32(sub32(mul32(tile_sizze_8998, tile_sizze_8998),
                                       local_tid_9421), group_sizze_8999), 1),
                     group_sizze_8999); i_9428++) {
            mem_9408[mul32(0, tile_sizze_8998)] = mem_9333[mul32(0,
                                                                 tile_sizze_8998)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        int32_t binop_x_9222 = mul32(tile_sizze_8998, num_whole_tiles_9008);
        int32_t ltid_x_9139 = squot32(local_tid_9421, tile_sizze_8998);
        int32_t ltid_y_9140;
        
        ltid_y_9140 = sub32(local_tid_9421, mul32(squot32(local_tid_9421,
                                                          tile_sizze_8998),
                                                  tile_sizze_8998));
        
        int32_t ltid_flat_9141;
        
        ltid_flat_9141 = local_tid_9421;
        if (slt32(ltid_x_9139, tile_sizze_8998) && slt32(ltid_y_9140,
                                                         tile_sizze_8998)) {
            int32_t i_9223 = add32(ltid_x_9139, binop_x_9222);
            int32_t j_9225 = add32(ltid_y_9140, binop_x_9222);
            int32_t gtid_9227 = add32(binop_x_9101, ltid_x_9139);
            int32_t gtid_9229 = add32(binop_x_9103, ltid_y_9140);
            bool binop_x_9232 = slt32(j_9225, m_8717);
            bool binop_y_9233 = slt32(gtid_9227, n_8716);
            bool cond_9234 = binop_x_9232 && binop_y_9233;
            double pre_9235;
            
            if (cond_9234) {
                double x_9236 = ((__global
                                  double *) xs_mem_9324)[add32(mul32(gtid_9227,
                                                                     m_8717),
                                                               j_9225)];
                
                pre_9235 = x_9236;
            } else {
                pre_9235 = 0.0;
            }
            
            bool binop_x_9238 = slt32(i_9223, m_8717);
            bool binop_y_9239 = slt32(gtid_9229, n_8716);
            bool cond_9240 = binop_x_9238 && binop_y_9239;
            double pre_9241;
            
            if (cond_9240) {
                double x_9242 = ((__global
                                  double *) mem_9329)[add32(mul32(i_9223,
                                                                  n_8716),
                                                            gtid_9229)];
                
                pre_9241 = x_9242;
            } else {
                pre_9241 = 0.0;
            }
            ((__local double *) mem_9353)[add32(mul32(ltid_x_9139,
                                                      tile_sizze_8998),
                                                ltid_y_9140)] = pre_9235;
            ((__local double *) mem_9358)[add32(mul32(ltid_x_9139,
                                                      tile_sizze_8998),
                                                ltid_y_9140)] = pre_9241;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int32_t ltid_x_9186 = squot32(local_tid_9421, tile_sizze_8998);
        int32_t ltid_y_9187;
        
        ltid_y_9187 = sub32(local_tid_9421, mul32(squot32(local_tid_9421,
                                                          tile_sizze_8998),
                                                  tile_sizze_8998));
        
        int32_t ltid_flat_9188;
        
        ltid_flat_9188 = local_tid_9421;
        if (slt32(ltid_x_9186, tile_sizze_8998) && slt32(ltid_y_9187,
                                                         tile_sizze_8998)) {
            int32_t gtid_9248 = add32(binop_x_9101, ltid_x_9186);
            int32_t gtid_9250 = add32(binop_x_9103, ltid_y_9187);
            double acc_9253 = mem_9333[mul32(0, tile_sizze_8998)];
            bool binop_x_9256 = slt32(gtid_9248, n_8716);
            bool binop_y_9257 = slt32(gtid_9250, n_8716);
            bool cond_9258 = binop_x_9256 && binop_y_9257;
            double acc_9259;
            
            if (cond_9258) {
                double x_9260;
                double redout_9298 = acc_9253;
                
                for (int32_t i_9299 = 0; i_9299 < residual_input_9137;
                     i_9299++) {
                    double x_9264 = ((__local
                                      double *) mem_9353)[add32(mul32(ltid_x_9186,
                                                                      tile_sizze_8998),
                                                                i_9299)];
                    double x_9265 = ((__local
                                      double *) mem_9358)[add32(mul32(i_9299,
                                                                      tile_sizze_8998),
                                                                ltid_y_9187)];
                    double res_9266 = x_9264 * x_9265;
                    double res_9263 = res_9266 + redout_9298;
                    double redout_tmp_9429 = res_9263;
                    
                    redout_9298 = redout_tmp_9429;
                }
                x_9260 = redout_9298;
                acc_9259 = x_9260;
            } else {
                acc_9259 = acc_9253;
            }
            mem_9362[mul32(0, tile_sizze_8998)] = acc_9259;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int32_t i_9430 = 0; i_9430 <
             squot32(sub32(add32(sub32(mul32(tile_sizze_8998, tile_sizze_8998),
                                       local_tid_9421), group_sizze_8999), 1),
                     group_sizze_8999); i_9430++) {
            mem_9408[mul32(0, tile_sizze_8998)] = mem_9362[mul32(0,
                                                                 tile_sizze_8998)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    double mem_9367[1];
    int32_t ltid_x_9268 = squot32(local_tid_9421, tile_sizze_8998);
    int32_t ltid_y_9269;
    
    ltid_y_9269 = sub32(local_tid_9421, mul32(squot32(local_tid_9421,
                                                      tile_sizze_8998),
                                              tile_sizze_8998));
    
    int32_t ltid_flat_9270;
    
    ltid_flat_9270 = local_tid_9421;
    if (slt32(ltid_x_9268, tile_sizze_8998) && slt32(ltid_y_9269,
                                                     tile_sizze_8998)) {
        int32_t gtid_9279 = add32(binop_x_9101, ltid_x_9268);
        int32_t gtid_9281 = add32(binop_x_9103, ltid_y_9269);
        bool binop_x_9283 = slt32(gtid_9279, n_8716);
        bool binop_y_9284 = slt32(gtid_9281, n_8716);
        bool cond_9285 = binop_x_9283 && binop_y_9284;
        double postlude_9286;
        
        if (cond_9285) {
            double res_9282 = mem_9408[mul32(0, tile_sizze_8998)];
            int8_t x_9289 = ((__global int8_t *) ys_mem_9325)[gtid_9279];
            int8_t x_9290 = ((__global int8_t *) ys_mem_9325)[gtid_9281];
            int8_t i8_arg_9291 = mul8(x_9289, x_9290);
            double res_9292 = sitofp_i8_f64(i8_arg_9291);
            double res_9293 = res_9282 * res_9292;
            
            postlude_9286 = res_9293;
        } else {
            postlude_9286 = 0.0;
        }
        mem_9367[mul32(0, tile_sizze_8998)] = postlude_9286;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int32_t thread_out_index_9431 = add32(mul32(gid_x_8995, tile_sizze_8998),
                                          squot32(local_tid_9421,
                                                  tile_sizze_8998));
    int32_t thread_out_index_9432;
    
    thread_out_index_9432 = add32(mul32(gid_y_8996, tile_sizze_8998),
                                  sub32(local_tid_9421,
                                        mul32(squot32(local_tid_9421,
                                                      tile_sizze_8998),
                                              tile_sizze_8998)));
    if (slt32(thread_out_index_9431, n_8716) && slt32(thread_out_index_9432,
                                                      n_8716)) {
        ((__global double *) mem_9372)[add32(mul32(thread_out_index_9431,
                                                   n_8716),
                                             thread_out_index_9432)] =
            mem_9367[mul32(0, tile_sizze_8998)];
    }
    
  error_6:
    return;
    #undef tile_sizze_8998
    #undef group_sizze_8999
}
"""
# Start of values.py.

# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        for c in read[::-1]:
            f.unget_char(c)
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in b'01234556789ABCDEFabcdef':
            s += c
            c = f.get_char()
        elif c == b'_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16)).encode('utf8') # ugh

def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in b'xX':
        c = f.get_char() # skip X
        return parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == b'_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
        if len(s) == 0:
            raise ValueError
        return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      return c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      return parse_int(f)

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    dims = []
    for i in range(rank):
        parse_specific_string(f, '[')
        dims += [int(parse_int(f))]
        parse_specific_string(f, ']')
    if np.product(dims) != 0:
        raise ValueError
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return tuple(dims)

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if type(elems) == tuple:
        # Empty array
        return np.empty(elems, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype=FUTHARK_PRIMTYPES[bin_type_enum]['numpy_type'])
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def end_of_input(entry, f=input_reader):
    skip_spaces(f)
    if f.get_char() != b'':
        panic(1, "Expected EOF on stdin after reading input for \"%s\".", entry)

def write_value_text(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[{}]'.format(d)
                                                    for d in v.shape]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

type_strs = { np.dtype('int8'): b'  i8',
              np.dtype('int16'): b' i16',
              np.dtype('int32'): b' i32',
              np.dtype('int64'): b' i64',
              np.dtype('uint8'): b'  u8',
              np.dtype('uint16'): b' u16',
              np.dtype('uint32'): b' u32',
              np.dtype('uint64'): b' u64',
              np.dtype('float32'): b' f32',
              np.dtype('float64'): b' f64',
              np.dtype('bool'): b'bool'}

def construct_binary_value(v):
    t = v.dtype
    shape = v.shape

    elems = 1
    for d in shape:
        elems *= d

    num_bytes = 1 + 1 + 1 + 4 + len(shape) * 8 + elems * t.itemsize
    bytes = bytearray(num_bytes)
    bytes[0] = np.int8(ord('b'))
    bytes[1] = 2
    bytes[2] = np.int8(len(shape))
    bytes[3:7] = type_strs[t]

    for i in range(len(shape)):
        bytes[7+i*8:7+(i+1)*8] = np.int64(shape[i]).tostring()

    bytes[7+len(shape)*8:] = np.ascontiguousarray(v).tostring()

    return bytes

def write_value_binary(v, out=sys.stdout):
    if sys.version_info >= (3,0):
        out = out.buffer
    out.write(construct_binary_value(v))

def write_value(v, out=sys.stdout, binary=False):
    if binary:
        return write_value_binary(v, out=out)
    else:
        return write_value_text(v, out=out)

# End of values.py.
# Start of memory.py.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, shape):
  # HACK: np.ctypeslib.as_array may fail if the shape contains zeroes,
  # for some reason.
  if any(map(lambda x: x == 0, shape)):
      return np.ndarray(shape, dtype=x._type_)
  else:
      return np.ctypeslib.as_array(x, shape=shape)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset*ct.sizeof(bt), bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset)*ct.sizeof(v), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)

# End of memory.py.
# Start of panic.py.

def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.stderr.write('\n')
    sys.exit(exitcode)

# End of panic.py.
# Start of tuning.py

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

# End of tuning.py.
# Start of scalar.py.

import numpy as np
import math
import struct

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def clz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if x < 0:
      break
    n += 1
    x <<= np.int8(1)
  return n

def popc_T(x):
  c = np.int32(0)
  while x != 0:
    x &= x - np.int8(1)
    c += np.int8(1)
  return c

futhark_popc8 = futhark_popc16 = futhark_popc32 = futhark_popc64 = popc_T
futhark_clzz8 = futhark_clzz16 = futhark_clzz32 = futhark_clzz64 = clz_T

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_mul_hi8(a, b):
  a = np.uint64(np.uint8(a))
  b = np.uint64(np.uint8(b))
  return np.int8((a*b) >> np.uint64(8))

def futhark_mul_hi16(a, b):
  a = np.uint64(np.uint16(a))
  b = np.uint64(np.uint16(b))
  return np.int16((a*b) >> np.uint64(16))

def futhark_mul_hi32(a, b):
  a = np.uint64(np.uint32(a))
  b = np.uint64(np.uint32(b))
  return np.int32((a*b) >> np.uint64(32))

# This one is done with arbitrary-precision integers.
def futhark_mul_hi64(a, b):
  a = int(np.uint64(a))
  b = int(np.uint64(b))
  return np.int64(np.uint64(a*b >> 64))

def futhark_mad_hi8(a, b, c):
  return futhark_mul_hi8(a,b) + c

def futhark_mad_hi16(a, b, c):
  return futhark_mul_hi16(a,b) + c

def futhark_mad_hi32(a, b, c):
  return futhark_mul_hi32(a,b) + c

def futhark_mad_hi64(a, b, c):
  return futhark_mul_hi64(a,b) + c

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_cosh64(x):
  return np.cosh(x)

def futhark_sinh64(x):
  return np.sinh(x)

def futhark_tanh64(x):
  return np.tanh(x)

def futhark_acosh64(x):
  return np.arccosh(x)

def futhark_asinh64(x):
  return np.arcsinh(x)

def futhark_atanh64(x):
  return np.arctanh(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_gamma64(x):
  return np.float64(math.gamma(x))

def futhark_lgamma64(x):
  return np.float64(math.lgamma(x))

def futhark_round64(x):
  return np.round(x)

def futhark_ceil64(x):
  return np.ceil(x)

def futhark_floor64(x):
  return np.floor(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_cosh32(x):
  return np.cosh(x)

def futhark_sinh32(x):
  return np.sinh(x)

def futhark_tanh32(x):
  return np.tanh(x)

def futhark_acosh32(x):
  return np.arccosh(x)

def futhark_asinh32(x):
  return np.arcsinh(x)

def futhark_atanh32(x):
  return np.arctanh(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_gamma32(x):
  return np.float32(math.gamma(x))

def futhark_lgamma32(x):
  return np.float32(math.lgamma(x))

def futhark_round32(x):
  return np.round(x)

def futhark_ceil32(x):
  return np.ceil(x)

def futhark_floor32(x):
  return np.floor(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])

def futhark_lerp32(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_lerp64(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_mad32(a, b, c):
  return a * b + c

def futhark_mad64(a, b, c):
  return a * b + c

def futhark_fma32(a, b, c):
  return a * b + c

def futhark_fma64(a, b, c):
  return a * b + c

# End of scalar.py.
runtime_file = None
do_warmup_run = False
num_runs = 1
entry_point = "main"
binary_output = False
parser = argparse.ArgumentParser(description="A compiled Futhark program.")
parser.add_argument("-t", "--write-runtime-to", action="append", default=[],
                    type=str)
parser.add_argument("-r", "--runs", action="append", default=[], type=str)
parser.add_argument("-e", "--entry-point", action="append", default=[],
                    type=str)
parser.add_argument("-b", "--binary-output", action="append_const", default=[],
                    const=None)
parser.add_argument("--tuning", action="append", default=[], type=open)
parser.add_argument("-p", "--platform", action="append", default=[], type=str)
parser.add_argument("-d", "--device", action="append", default=[], type=str)
parser.add_argument("--default-threshold", action="append", default=[],
                    type=int)
parser.add_argument("--default-group-size", action="append", default=[],
                    type=int)
parser.add_argument("--default-num-groups", action="append", default=[],
                    type=int)
parser.add_argument("--default-tile-size", action="append", default=[],
                    type=int)
parser.add_argument("--size", action="append", default=[], type=size_assignment)
parser_result = vars(parser.parse_args(sys.argv[1:]))
for optarg in parser_result["write_runtime_to"]:
  if runtime_file:
    runtime_file.close()
  runtime_file = open(optarg, "w")
for optarg in parser_result["runs"]:
  num_runs = optarg
  do_warmup_run = True
for optarg in parser_result["entry_point"]:
  entry_point = optarg
for optarg in parser_result["binary_output"]:
  binary_output = True
for optarg in parser_result["tuning"]:
  read_tuning_file(sizes, optarg)
for optarg in parser_result["platform"]:
  preferred_platform = optarg
for optarg in parser_result["device"]:
  preferred_device = optarg
for optarg in parser_result["default_threshold"]:
  default_threshold = optarg
for optarg in parser_result["default_group_size"]:
  default_group_size = optarg
for optarg in parser_result["default_num_groups"]:
  default_num_groups = optarg
for optarg in parser_result["default_tile_size"]:
  default_tile_size = optarg
for optarg in parser_result["size"]:
  sizes[optarg[0]] = optarg[1]
class internal:
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width", 32),
     ("AMD Accelerated Parallel Processing", cl.device_type.GPU, "lockstep_width",
      32), ("", cl.device_type.GPU, "lockstep_width", 1), ("", cl.device_type.GPU,
                                                           "num_groups", 256), ("",
                                                                                cl.device_type.GPU,
                                                                                "group_size",
                                                                                256),
     ("", cl.device_type.GPU, "tile_size", 32), ("", cl.device_type.GPU,
                                                 "threshold", 32768), ("",
                                                                       cl.device_type.CPU,
                                                                       "lockstep_width",
                                                                       1), ("",
                                                                            cl.device_type.CPU,
                                                                            "num_groups",
                                                                            "MAX_COMPUTE_UNITS"),
     ("", cl.device_type.CPU, "group_size", 32), ("", cl.device_type.CPU,
                                                  "tile_size", 4), ("",
                                                                    cl.device_type.CPU,
                                                                    "threshold",
                                                                    "MAX_COMPUTE_UNITS")]
    self.global_failure_args_max = 0
    self.failure_msgs=[]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i8", "i32", "f64", "bool"],
                                       user_sizes=sizes,
                                       all_sizes={"train_binary.group_size_9440": {"class": "group_size", "value": None},
                                        "train_binary.segmap_group_size_8959": {"class": "group_size", "value": None},
                                        "train_binary.segmap_group_size_8979": {"class": "group_size", "value": None},
                                        "train_binary.tile_size_8997": {"class": "tile_size", "value": None}})
    self.map_transpose_f64_var = program.map_transpose_f64
    self.map_transpose_f64_low_height_var = program.map_transpose_f64_low_height
    self.map_transpose_f64_low_width_var = program.map_transpose_f64_low_width
    self.map_transpose_f64_small_var = program.map_transpose_f64_small
    self.replicate_9437_var = program.replicate_9437
    self.segmap_8956_var = program.segmap_8956
    self.segmap_8976_var = program.segmap_8976
    self.segmap_intragroup_9007_var = program.segmap_intragroup_9007
  def futhark_train_binary(self, xs_mem_9324, ys_mem_9325, n_8716, m_8717,
                           n_8718):
    dim_match_8721 = (n_8716 == n_8718)
    empty_or_match_cert_8722 = True
    assert dim_match_8721, ("Error: %s\n\nBacktrace:\n-> #0  smo2.fut:10:1-83:6\n" % ("function arguments of wrong shape",))
    n_8929 = sext_i32_i64(n_8716)
    convop_x_9327 = (n_8716 * m_8717)
    binop_x_9328 = sext_i32_i64(convop_x_9327)
    bytes_9326 = (np.int64(8) * binop_x_9328)
    mem_9329 = opencl_alloc(self, bytes_9326, "mem_9329")
    self.futhark__map_transpose_f64(mem_9329, np.int32(0), xs_mem_9324,
                                    np.int32(0), np.int32(1), m_8717, n_8716,
                                    (n_8716 * m_8717), (n_8716 * m_8717))
    tile_sizze_8998 = self.sizes["train_binary.tile_size_8997"]
    group_sizze_8999 = (tile_sizze_8998 * tile_sizze_8998)
    y_9000 = (tile_sizze_8998 - np.int32(1))
    x_9001 = (n_8716 + y_9000)
    num_groups_x_9002 = squot32(x_9001, tile_sizze_8998)
    num_groups_top_9006 = (num_groups_x_9002 * num_groups_x_9002)
    num_whole_tiles_9008 = squot32(m_8717, tile_sizze_8998)
    residual_input_9137 = srem32(m_8717, tile_sizze_8998)
    cond_9138 = (residual_input_9137 == np.int32(0))
    binop_x_9371 = (n_8929 * n_8929)
    bytes_9368 = (np.int64(8) * binop_x_9371)
    mem_9372 = opencl_alloc(self, bytes_9368, "mem_9372")
    binop_x_9332 = sext_i32_i64(group_sizze_8999)
    bytes_9330 = (np.int64(8) * binop_x_9332)
    binop_x_9335 = sext_i32_i64(tile_sizze_8998)
    binop_x_9337 = (binop_x_9335 * binop_x_9335)
    bytes_9334 = (np.int64(8) * binop_x_9337)
    if ((1 * (np.long(num_groups_top_9006) * np.long(group_sizze_8999))) != 0):
      self.segmap_intragroup_9007_var.set_args(self.global_failure,
                                               cl.LocalMemory(np.long(bytes_9334)),
                                               cl.LocalMemory(np.long(bytes_9334)),
                                               cl.LocalMemory(np.long(bytes_9334)),
                                               cl.LocalMemory(np.long(bytes_9334)),
                                               np.int32(n_8716),
                                               np.int32(m_8717),
                                               np.int32(num_groups_x_9002),
                                               np.int32(num_whole_tiles_9008),
                                               np.int32(residual_input_9137),
                                               np.byte(cond_9138), xs_mem_9324,
                                               ys_mem_9325, mem_9329, mem_9372)
      cl.enqueue_nd_range_kernel(self.queue, self.segmap_intragroup_9007_var,
                                 ((np.long(num_groups_top_9006) * np.long(group_sizze_8999)),),
                                 (np.long(group_sizze_8999),))
      if synchronous:
        sync(self)
    mem_9329 = None
    bytes_9373 = (np.int64(8) * n_8929)
    mem_9375 = opencl_alloc(self, bytes_9373, "mem_9375")
    self.futhark__replicate_f64(mem_9375, n_8716, np.float64(0.0))
    mem_9378 = opencl_alloc(self, bytes_9373, "mem_9378")
    self.futhark__replicate_f64(mem_9378, n_8716, np.float64(-1.0))
    mem_9383 = opencl_alloc(self, np.int64(8), "mem_9383")
    A_mem_9379 = mem_9375
    G_mem_9380 = mem_9378
    loop_while_8748 = True
    while loop_while_8748:
      i_8753 = np.int32(-1)
      G_max_8754 = -np.inf
      t_8755 = np.int32(0)
      one_9466 = np.int32(1)
      for counter_9465 in range(n_8716):
        read_res_9462 = np.empty(1, dtype=ct.c_int8)
        cl.enqueue_copy(self.queue, read_res_9462, ys_mem_9325,
                        device_offset=(np.long(t_8755) * 1),
                        is_blocking=synchronous)
        sync(self)
        y_t_8760 = read_res_9462[0]
        read_res_9463 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9463, A_mem_9379,
                        device_offset=(np.long(t_8755) * 8),
                        is_blocking=synchronous)
        sync(self)
        A_t_8761 = read_res_9463[0]
        res_8762 = sitofp_i8_f64(y_t_8760)
        cond_8763 = (y_t_8760 == np.int8(1))
        res_8764 = (A_t_8761 < np.float64(10.0))
        x_8765 = (cond_8763 and res_8764)
        cond_8766 = (y_t_8760 == np.int8(-1))
        res_8767 = (np.float64(0.0) < A_t_8761)
        x_8768 = (cond_8766 and res_8767)
        x_8769 = not(x_8765)
        y_8770 = (x_8768 and x_8769)
        cond0_8771 = (x_8765 or y_8770)
        read_res_9464 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9464, G_mem_9380,
                        device_offset=(np.long(t_8755) * 8),
                        is_blocking=synchronous)
        sync(self)
        y_8772 = read_res_9464[0]
        negate_arg_8773 = (res_8762 * y_8772)
        x_8774 = (np.float64(0.0) - negate_arg_8773)
        cond1_8775 = (G_max_8754 <= x_8774)
        x_8776 = (cond0_8771 and cond1_8775)
        if x_8776:
          loopres_8777 = t_8755
        else:
          loopres_8777 = i_8753
        if x_8776:
          loopres_8778 = x_8774
        else:
          loopres_8778 = G_max_8754
        i_tmp_9447 = loopres_8777
        G_max_tmp_9448 = loopres_8778
        i_8753 = i_tmp_9447
        G_max_8754 = G_max_tmp_9448
        t_8755 += one_9466
      loopres_8751 = i_8753
      loopres_8752 = G_max_8754
      x_8779 = sle32(np.int32(0), loopres_8751)
      y_8780 = slt32(loopres_8751, n_8716)
      bounds_check_8781 = (x_8779 and y_8780)
      index_certs_8782 = True
      assert bounds_check_8781, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:38:23-27\n   #1  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                             loopres_8751,
                                                                                                                             "] out of bounds for array of shape [",
                                                                                                                             n_8716,
                                                                                                                             "]."))
      read_res_9467 = np.empty(1, dtype=ct.c_int8)
      cl.enqueue_copy(self.queue, read_res_9467, ys_mem_9325,
                      device_offset=(np.long(loopres_8751) * 1),
                      is_blocking=synchronous)
      sync(self)
      i8_arg_8783 = read_res_9467[0]
      res_8784 = sitofp_i8_f64(i8_arg_8783)
      j_8788 = np.int32(-1)
      obj_min_8789 = np.inf
      G_min_8790 = np.inf
      t_8791 = np.int32(0)
      one_9475 = np.int32(1)
      for counter_9474 in range(n_8716):
        read_res_9468 = np.empty(1, dtype=ct.c_int8)
        cl.enqueue_copy(self.queue, read_res_9468, ys_mem_9325,
                        device_offset=(np.long(t_8791) * 1),
                        is_blocking=synchronous)
        sync(self)
        y_t_8796 = read_res_9468[0]
        read_res_9469 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9469, A_mem_9379,
                        device_offset=(np.long(t_8791) * 8),
                        is_blocking=synchronous)
        sync(self)
        A_t_8797 = read_res_9469[0]
        res_8798 = sitofp_i8_f64(y_t_8796)
        cond_8799 = (y_t_8796 == np.int8(1))
        res_8800 = (np.float64(0.0) < A_t_8797)
        x_8801 = (cond_8799 and res_8800)
        cond_8802 = (y_t_8796 == np.int8(-1))
        res_8803 = (A_t_8797 < np.float64(10.0))
        x_8804 = (cond_8802 and res_8803)
        x_8805 = not(x_8801)
        y_8806 = (x_8804 and x_8805)
        cond0_8807 = (x_8801 or y_8806)
        x_8808 = (np.float64(0.0) - res_8798)
        read_res_9470 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9470, G_mem_9380,
                        device_offset=(np.long(t_8791) * 8),
                        is_blocking=synchronous)
        sync(self)
        y_8809 = read_res_9470[0]
        x_8810 = (x_8808 * y_8809)
        cond1_8811 = (x_8810 <= G_min_8790)
        cond_8812 = not(cond0_8807)
        if cond_8812:
          loopres_8813 = j_8788
          loopres_8814 = obj_min_8789
          loopres_8815 = G_min_8790
        else:
          y_8816 = (res_8798 * y_8809)
          b_8817 = (loopres_8752 + y_8816)
          if cond1_8811:
            res_8819 = (np.float64(0.0) - y_8816)
            G_minzq_8818 = res_8819
          else:
            G_minzq_8818 = G_min_8790
          complement_arg_8820 = (np.float64(0.0) < b_8817)
          cond_8821 = not(complement_arg_8820)
          if cond_8821:
            res_8822 = j_8788
            res_8823 = obj_min_8789
          else:
            index_ok_8824 = (bounds_check_8781 and bounds_check_8781)
            index_certs_8825 = True
            assert index_ok_8824, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:50:15-21\n   #1  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                                       loopres_8751,
                                                                                                                                       ", ",
                                                                                                                                       loopres_8751,
                                                                                                                                       "] out of bounds for array of shape [",
                                                                                                                                       n_8716,
                                                                                                                                       "][",
                                                                                                                                       n_8716,
                                                                                                                                       "]."))
            read_res_9471 = np.empty(1, dtype=ct.c_double)
            cl.enqueue_copy(self.queue, read_res_9471, mem_9372,
                            device_offset=(np.long(((loopres_8751 * n_8716) + loopres_8751)) * 8),
                            is_blocking=synchronous)
            sync(self)
            x_8826 = read_res_9471[0]
            read_res_9472 = np.empty(1, dtype=ct.c_double)
            cl.enqueue_copy(self.queue, read_res_9472, mem_9372,
                            device_offset=(np.long(((t_8791 * n_8716) + t_8791)) * 8),
                            is_blocking=synchronous)
            sync(self)
            y_8829 = read_res_9472[0]
            x_8830 = (x_8826 + y_8829)
            x_8831 = (np.float64(2.0) * res_8784)
            x_8832 = (res_8798 * x_8831)
            index_certs_8834 = True
            assert bounds_check_8781, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:50:56-62\n   #1  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                                           loopres_8751,
                                                                                                                                           ", ",
                                                                                                                                           t_8791,
                                                                                                                                           "] out of bounds for array of shape [",
                                                                                                                                           n_8716,
                                                                                                                                           "][",
                                                                                                                                           n_8716,
                                                                                                                                           "]."))
            read_res_9473 = np.empty(1, dtype=ct.c_double)
            cl.enqueue_copy(self.queue, read_res_9473, mem_9372,
                            device_offset=(np.long(((loopres_8751 * n_8716) + t_8791)) * 8),
                            is_blocking=synchronous)
            sync(self)
            y_8835 = read_res_9473[0]
            y_8836 = (x_8832 * y_8835)
            a_8837 = (x_8830 - y_8836)
            cond_8838 = (a_8837 <= np.float64(0.0))
            if cond_8838:
              a_8839 = np.float64(1.0e-6)
            else:
              a_8839 = a_8837
            x_8840 = (b_8817 * b_8817)
            negate_arg_8841 = (x_8840 / a_8839)
            x_8842 = (np.float64(0.0) - negate_arg_8841)
            cond2_8843 = (x_8842 <= obj_min_8789)
            if cond2_8843:
              res_8844 = t_8791
            else:
              res_8844 = j_8788
            if cond2_8843:
              res_8845 = x_8842
            else:
              res_8845 = obj_min_8789
            res_8822 = res_8844
            res_8823 = res_8845
          loopres_8813 = res_8822
          loopres_8814 = res_8823
          loopres_8815 = G_minzq_8818
        j_tmp_9449 = loopres_8813
        obj_min_tmp_9450 = loopres_8814
        G_min_tmp_9451 = loopres_8815
        j_8788 = j_tmp_9449
        obj_min_8789 = obj_min_tmp_9450
        G_min_8790 = G_min_tmp_9451
        t_8791 += one_9475
      loopres_8785 = j_8788
      loopres_8786 = obj_min_8789
      loopres_8787 = G_min_8790
      x_8846 = (loopres_8752 - loopres_8787)
      cond0_8847 = (x_8846 < np.float64(1.0e-3))
      res_8848 = (loopres_8785 == np.int32(-1))
      x_8849 = not(cond0_8847)
      y_8850 = (res_8848 and x_8849)
      cond_8851 = (cond0_8847 or y_8850)
      cl.enqueue_copy(self.queue, mem_9383, np.array(loopres_8751,
                                                     dtype=ct.c_int32),
                      device_offset=(np.long(np.int32(0)) * 4),
                      is_blocking=synchronous)
      cl.enqueue_copy(self.queue, mem_9383, np.array(loopres_8785,
                                                     dtype=ct.c_int32),
                      device_offset=(np.long(np.int32(1)) * 4),
                      is_blocking=synchronous)
      x_8853 = not(cond_8851)
      if cond_8851:
        loopres_mem_9393 = A_mem_9379
        loopres_mem_9394 = G_mem_9380
      else:
        index_ok_8856 = (bounds_check_8781 and bounds_check_8781)
        index_certs_8857 = True
        assert index_ok_8856, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:59:13-19\n   #1  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                                   loopres_8751,
                                                                                                                                   ", ",
                                                                                                                                   loopres_8751,
                                                                                                                                   "] out of bounds for array of shape [",
                                                                                                                                   n_8716,
                                                                                                                                   "][",
                                                                                                                                   n_8716,
                                                                                                                                   "]."))
        read_res_9476 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9476, mem_9372,
                        device_offset=(np.long(((loopres_8751 * n_8716) + loopres_8751)) * 8),
                        is_blocking=synchronous)
        sync(self)
        x_8858 = read_res_9476[0]
        x_8859 = sle32(np.int32(0), loopres_8785)
        y_8860 = slt32(loopres_8785, n_8716)
        bounds_check_8861 = (x_8859 and y_8860)
        index_ok_8862 = (bounds_check_8861 and bounds_check_8861)
        index_certs_8863 = True
        assert index_ok_8862, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:59:23-29\n   #1  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                                   loopres_8785,
                                                                                                                                   ", ",
                                                                                                                                   loopres_8785,
                                                                                                                                   "] out of bounds for array of shape [",
                                                                                                                                   n_8716,
                                                                                                                                   "][",
                                                                                                                                   n_8716,
                                                                                                                                   "]."))
        read_res_9477 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9477, mem_9372,
                        device_offset=(np.long(((loopres_8785 * n_8716) + loopres_8785)) * 8),
                        is_blocking=synchronous)
        sync(self)
        y_8864 = read_res_9477[0]
        x_8865 = (x_8858 + y_8864)
        index_certs_8866 = True
        assert bounds_check_8861, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:59:53-57\n   #1  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                               loopres_8785,
                                                                                                                               "] out of bounds for array of shape [",
                                                                                                                               n_8716,
                                                                                                                               "]."))
        read_res_9478 = np.empty(1, dtype=ct.c_int8)
        cl.enqueue_copy(self.queue, read_res_9478, ys_mem_9325,
                        device_offset=(np.long(loopres_8785) * 1),
                        is_blocking=synchronous)
        sync(self)
        y_8867 = read_res_9478[0]
        i8_arg_8868 = (i8_arg_8783 * y_8867)
        res_8869 = sitofp_i8_f64(i8_arg_8868)
        x_8870 = (np.float64(2.0) * res_8869)
        index_ok_8871 = (bounds_check_8781 and bounds_check_8861)
        index_certs_8872 = True
        assert index_ok_8871, ("Error: %s%d%s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:59:62-68\n   #1  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                                   loopres_8751,
                                                                                                                                   ", ",
                                                                                                                                   loopres_8785,
                                                                                                                                   "] out of bounds for array of shape [",
                                                                                                                                   n_8716,
                                                                                                                                   "][",
                                                                                                                                   n_8716,
                                                                                                                                   "]."))
        read_res_9479 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9479, mem_9372,
                        device_offset=(np.long(((loopres_8751 * n_8716) + loopres_8785)) * 8),
                        is_blocking=synchronous)
        sync(self)
        y_8873 = read_res_9479[0]
        y_8874 = (x_8870 * y_8873)
        a_8875 = (x_8865 - y_8874)
        cond_8876 = (a_8875 <= np.float64(0.0))
        if cond_8876:
          a_8877 = np.float64(1.0e-6)
        else:
          a_8877 = a_8875
        i8_arg_8878 = (np.int8(0) - i8_arg_8783)
        res_8879 = sitofp_i8_f64(i8_arg_8878)
        read_res_9480 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9480, G_mem_9380,
                        device_offset=(np.long(loopres_8751) * 8),
                        is_blocking=synchronous)
        sync(self)
        y_8880 = read_res_9480[0]
        x_8881 = (res_8879 * y_8880)
        res_8882 = sitofp_i8_f64(y_8867)
        read_res_9481 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9481, G_mem_9380,
                        device_offset=(np.long(loopres_8785) * 8),
                        is_blocking=synchronous)
        sync(self)
        y_8883 = read_res_9481[0]
        y_8884 = (res_8882 * y_8883)
        b_8885 = (x_8881 + y_8884)
        read_res_9482 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9482, A_mem_9379,
                        device_offset=(np.long(loopres_8751) * 8),
                        is_blocking=synchronous)
        sync(self)
        x_8886 = read_res_9482[0]
        x_8887 = (res_8784 * b_8885)
        y_8888 = (x_8887 / a_8877)
        A_i_8889 = (x_8886 + y_8888)
        x_8890 = (res_8784 * x_8886)
        read_res_9483 = np.empty(1, dtype=ct.c_double)
        cl.enqueue_copy(self.queue, read_res_9483, A_mem_9379,
                        device_offset=(np.long(loopres_8785) * 8),
                        is_blocking=synchronous)
        sync(self)
        y_8891 = read_res_9483[0]
        y_8892 = (res_8882 * y_8891)
        sum_8893 = (x_8890 + y_8892)
        cond_8894 = (A_i_8889 < np.float64(0.0))
        if cond_8894:
          res_8895 = np.float64(0.0)
        else:
          cond_8896 = (np.float64(10.0) < A_i_8889)
          if cond_8896:
            res_8897 = np.float64(10.0)
          else:
            res_8897 = A_i_8889
          res_8895 = res_8897
        y_8898 = (res_8784 * res_8895)
        y_8899 = (sum_8893 - y_8898)
        A_j_8900 = (res_8882 * y_8899)
        cond_8901 = (A_j_8900 < np.float64(0.0))
        if cond_8901:
          res_8902 = np.float64(0.0)
        else:
          cond_8903 = (np.float64(10.0) < A_j_8900)
          if cond_8903:
            res_8904 = np.float64(10.0)
          else:
            res_8904 = A_j_8900
          res_8902 = res_8904
        y_8905 = (res_8882 * res_8902)
        y_8906 = (sum_8893 - y_8905)
        A_i_8907 = (res_8784 * y_8906)
        deltaAi_8908 = (A_i_8907 - x_8886)
        deltaAj_8909 = (res_8902 - y_8891)
        index_certs_8910 = True
        assert bounds_check_8781, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:77:33-36\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  smo2.fut:77:14-68\n   #4  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                                                                                                                                loopres_8751,
                                                                                                                                                                                                                                "] out of bounds for array of shape [",
                                                                                                                                                                                                                                n_8716,
                                                                                                                                                                                                                                "]."))
        index_certs_8911 = True
        assert bounds_check_8861, ("Error: %s%d%s%d%s\n\nBacktrace:\n-> #0  smo2.fut:77:50-53\n   #1  /prelude/soacs.fut:56:19-23\n   #2  /prelude/soacs.fut:56:3-37\n   #3  smo2.fut:77:14-68\n   #4  smo2.fut:10:1-83:6\n" % ("Index [",
                                                                                                                                                                                                                                loopres_8785,
                                                                                                                                                                                                                                "] out of bounds for array of shape [",
                                                                                                                                                                                                                                n_8716,
                                                                                                                                                                                                                                "]."))
        mem_9386 = opencl_alloc(self, np.int64(16), "mem_9386")
        cl.enqueue_copy(self.queue, mem_9386, np.array(A_i_8907,
                                                       dtype=ct.c_double),
                        device_offset=(np.long(np.int32(0)) * 8),
                        is_blocking=synchronous)
        cl.enqueue_copy(self.queue, mem_9386, np.array(res_8902,
                                                       dtype=ct.c_double),
                        device_offset=(np.long(np.int32(1)) * 8),
                        is_blocking=synchronous)
        mem_9389 = opencl_alloc(self, bytes_9373, "mem_9389")
        if ((sext_i32_i64(n_8716) * np.int32(8)) != 0):
          cl.enqueue_copy(self.queue, mem_9389, A_mem_9379,
                          dest_offset=np.long(np.int32(0)),
                          src_offset=np.long(np.int32(0)),
                          byte_count=np.long((sext_i32_i64(n_8716) * np.int32(8))))
        if synchronous:
          sync(self)
        segmap_group_sizze_8960 = self.sizes["train_binary.segmap_group_size_8959"]
        segmap_group_sizze_8961 = sext_i32_i64(segmap_group_sizze_8960)
        y_8962 = (segmap_group_sizze_8961 - np.int64(1))
        x_8963 = (n_8929 + y_8962)
        segmap_usable_groups_64_8965 = squot64(x_8963, segmap_group_sizze_8961)
        segmap_usable_groups_8966 = sext_i64_i32(segmap_usable_groups_64_8965)
        mem_9392 = opencl_alloc(self, bytes_9373, "mem_9392")
        if ((1 * (np.long(segmap_usable_groups_8966) * np.long(segmap_group_sizze_8960))) != 0):
          self.segmap_8956_var.set_args(self.global_failure, np.int32(n_8716),
                                        np.int32(loopres_8751),
                                        np.int32(loopres_8785),
                                        np.float64(deltaAi_8908),
                                        np.float64(deltaAj_8909), mem_9372,
                                        G_mem_9380, mem_9392)
          cl.enqueue_nd_range_kernel(self.queue, self.segmap_8956_var,
                                     ((np.long(segmap_usable_groups_8966) * np.long(segmap_group_sizze_8960)),),
                                     (np.long(segmap_group_sizze_8960),))
          if synchronous:
            sync(self)
        segmap_group_sizze_8980 = self.sizes["train_binary.segmap_group_size_8979"]
        segmap_group_sizze_8981 = sext_i32_i64(segmap_group_sizze_8980)
        y_8982 = (segmap_group_sizze_8981 - np.int64(1))
        x_8983 = (np.int64(2) + y_8982)
        segmap_usable_groups_64_8985 = squot64(x_8983, segmap_group_sizze_8981)
        segmap_usable_groups_8986 = sext_i64_i32(segmap_usable_groups_64_8985)
        if ((1 * (np.long(segmap_usable_groups_8986) * np.long(segmap_group_sizze_8980))) != 0):
          self.segmap_8976_var.set_args(self.global_failure, np.int32(n_8716),
                                        mem_9383, mem_9386, mem_9389)
          cl.enqueue_nd_range_kernel(self.queue, self.segmap_8976_var,
                                     ((np.long(segmap_usable_groups_8986) * np.long(segmap_group_sizze_8980)),),
                                     (np.long(segmap_group_sizze_8980),))
          if synchronous:
            sync(self)
        mem_9386 = None
        loopres_mem_9393 = mem_9389
        loopres_mem_9394 = mem_9392
      A_mem_tmp_9442 = loopres_mem_9393
      G_mem_tmp_9443 = loopres_mem_9394
      loop_while_tmp_9444 = x_8853
      A_mem_9379 = A_mem_tmp_9442
      G_mem_9380 = G_mem_tmp_9443
      loop_while_8748 = loop_while_tmp_9444
    A_mem_9395 = A_mem_9379
    A_mem_9396 = G_mem_9380
    A_8745 = loop_while_8748
    mem_9372 = None
    mem_9375 = None
    mem_9378 = None
    mem_9383 = None
    out_arrsizze_9419 = n_8716
    out_mem_9418 = A_mem_9395
    return (out_mem_9418, out_arrsizze_9419)
  def futhark__replicate_f64(self, mem_9433, num_elems_9434, val_9435):
    group_sizze_9440 = self.sizes["train_binary.group_size_9440"]
    num_groups_9441 = squot32(((num_elems_9434 + sext_i32_i32(group_sizze_9440)) - np.int32(1)),
                              sext_i32_i32(group_sizze_9440))
    if ((1 * (np.long(num_groups_9441) * np.long(group_sizze_9440))) != 0):
      self.replicate_9437_var.set_args(mem_9433, np.int32(num_elems_9434),
                                       np.float64(val_9435))
      cl.enqueue_nd_range_kernel(self.queue, self.replicate_9437_var,
                                 ((np.long(num_groups_9441) * np.long(group_sizze_9440)),),
                                 (np.long(group_sizze_9440),))
      if synchronous:
        sync(self)
    return ()
  def futhark__map_transpose_f64(self, destmem_0, destoffset_1, srcmem_2,
                                 srcoffset_3, num_arrays_4, x_elems_5,
                                 y_elems_6, in_elems_7, out_elems_8):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_10 = squot32(np.int32(16), x_elems_5)
      mulx_9 = squot32(np.int32(16), y_elems_6)
      if ((in_elems_7 == out_elems_8) and (((num_arrays_4 == np.int32(1)) or ((x_elems_5 * y_elems_6) == in_elems_7)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1))))):
        if ((in_elems_7 * np.int32(8)) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(destoffset_1),
                          src_offset=np.long(srcoffset_3),
                          byte_count=np.long((in_elems_7 * np.int32(8))))
        if synchronous:
          sync(self)
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                                                            muly_10) + np.int32(16)) - np.int32(1)),
                                                                                                  np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.map_transpose_f64_low_width_var.set_args(cl.LocalMemory(np.long(np.int32(2176))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.map_transpose_f64_low_width_var,
                                       ((np.long(squot32(((x_elems_5 + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(squot32(((squot32(((y_elems_6 + muly_10) - np.int32(1)),
                                                                   muly_10) + np.int32(16)) - np.int32(1)),
                                                         np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              sync(self)
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                  mulx_9) + np.int32(16)) - np.int32(1)),
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                                                                    np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.map_transpose_f64_low_height_var.set_args(cl.LocalMemory(np.long(np.int32(2176))),
                                                             np.int32(destoffset_1),
                                                             np.int32(srcoffset_3),
                                                             np.int32(num_arrays_4),
                                                             np.int32(x_elems_5),
                                                             np.int32(y_elems_6),
                                                             np.int32(in_elems_7),
                                                             np.int32(out_elems_8),
                                                             np.int32(mulx_9),
                                                             np.int32(muly_10),
                                                             destmem_0,
                                                             srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.map_transpose_f64_low_height_var,
                                         ((np.long(squot32(((squot32(((x_elems_5 + mulx_9) - np.int32(1)),
                                                                     mulx_9) + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(squot32(((y_elems_6 + np.int32(16)) - np.int32(1)),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                sync(self)
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                        np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.map_transpose_f64_small_var.set_args(cl.LocalMemory(np.long(np.int32(1))),
                                                          np.int32(destoffset_1),
                                                          np.int32(srcoffset_3),
                                                          np.int32(num_arrays_4),
                                                          np.int32(x_elems_5),
                                                          np.int32(y_elems_6),
                                                          np.int32(in_elems_7),
                                                          np.int32(out_elems_8),
                                                          np.int32(mulx_9),
                                                          np.int32(muly_10),
                                                          destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f64_small_var,
                                           ((np.long(squot32(((((num_arrays_4 * x_elems_5) * y_elems_6) + np.int32(256)) - np.int32(1)),
                                                             np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  sync(self)
            else:
              if ((((1 * (np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                          np.int32(32))) * np.long(np.int32(32)))) * (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                                                                      np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.map_transpose_f64_var.set_args(cl.LocalMemory(np.long(np.int32(8448))),
                                                    np.int32(destoffset_1),
                                                    np.int32(srcoffset_3),
                                                    np.int32(num_arrays_4),
                                                    np.int32(x_elems_5),
                                                    np.int32(y_elems_6),
                                                    np.int32(in_elems_7),
                                                    np.int32(out_elems_8),
                                                    np.int32(mulx_9),
                                                    np.int32(muly_10),
                                                    destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.map_transpose_f64_var,
                                           ((np.long(squot32(((x_elems_5 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(squot32(((y_elems_6 + np.int32(32)) - np.int32(1)),
                                                             np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  sync(self)
    return ()
self = internal()
def entry_train_binary():
  xs_mem_9324_ext = read_value("[][]f64")
  ys_mem_9325_ext = read_value("[]i8")
  end_of_input("train_binary")
  try:
    assert ((type(xs_mem_9324_ext) in [np.ndarray,
                                       cl.array.Array]) and (xs_mem_9324_ext.dtype == np.float64)), "Parameter has unexpected type"
    n_8716 = np.int32(xs_mem_9324_ext.shape[0])
    m_8717 = np.int32(xs_mem_9324_ext.shape[1])
    if (type(xs_mem_9324_ext) == cl.array.Array):
      xs_mem_9324 = xs_mem_9324_ext.data
    else:
      xs_mem_9324 = opencl_alloc(self, np.int64(xs_mem_9324_ext.nbytes),
                                 "xs_mem_9324")
      if (np.int64(xs_mem_9324_ext.nbytes) != 0):
        cl.enqueue_copy(self.queue, xs_mem_9324,
                        normaliseArray(xs_mem_9324_ext),
                        is_blocking=synchronous)
  except (TypeError, AssertionError) as e:
    raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                          type(xs_mem_9324_ext),
                                                                                                                          xs_mem_9324_ext))
  try:
    assert ((type(ys_mem_9325_ext) in [np.ndarray,
                                       cl.array.Array]) and (ys_mem_9325_ext.dtype == np.int8)), "Parameter has unexpected type"
    n_8718 = np.int32(ys_mem_9325_ext.shape[0])
    if (type(ys_mem_9325_ext) == cl.array.Array):
      ys_mem_9325 = ys_mem_9325_ext.data
    else:
      ys_mem_9325 = opencl_alloc(self, np.int64(ys_mem_9325_ext.nbytes),
                                 "ys_mem_9325")
      if (np.int64(ys_mem_9325_ext.nbytes) != 0):
        cl.enqueue_copy(self.queue, ys_mem_9325,
                        normaliseArray(ys_mem_9325_ext),
                        is_blocking=synchronous)
  except (TypeError, AssertionError) as e:
    raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[]i8",
                                                                                                                          type(ys_mem_9325_ext),
                                                                                                                          ys_mem_9325_ext))
  try:
    with np.errstate(divide="ignore", over="ignore", under="ignore",
                     invalid="ignore"):
      if do_warmup_run:
        xs_mem_copy_9484 = opencl_alloc(self, xs_mem_9324_ext.nbytes,
                                        "xs_mem_copy_9484")
        if (xs_mem_9324_ext.nbytes != 0):
          cl.enqueue_copy(self.queue, xs_mem_copy_9484, xs_mem_9324,
                          dest_offset=np.long(0), src_offset=np.long(0),
                          byte_count=np.long(xs_mem_9324_ext.nbytes))
        if synchronous:
          sync(self)
        ys_mem_copy_9485 = opencl_alloc(self, ys_mem_9325_ext.nbytes,
                                        "ys_mem_copy_9485")
        if (ys_mem_9325_ext.nbytes != 0):
          cl.enqueue_copy(self.queue, ys_mem_copy_9485, ys_mem_9325,
                          dest_offset=np.long(0), src_offset=np.long(0),
                          byte_count=np.long(ys_mem_9325_ext.nbytes))
        if synchronous:
          sync(self)
        (out_mem_9418,
         out_arrsizze_9419) = self.futhark_train_binary(xs_mem_copy_9484,
                                                        ys_mem_copy_9485,
                                                        n_8716, m_8717, n_8718)
        sync(self)
      for i in range(int(num_runs)):
        xs_mem_copy_9484 = opencl_alloc(self, xs_mem_9324_ext.nbytes,
                                        "xs_mem_copy_9484")
        if (xs_mem_9324_ext.nbytes != 0):
          cl.enqueue_copy(self.queue, xs_mem_copy_9484, xs_mem_9324,
                          dest_offset=np.long(0), src_offset=np.long(0),
                          byte_count=np.long(xs_mem_9324_ext.nbytes))
        if synchronous:
          sync(self)
        ys_mem_copy_9485 = opencl_alloc(self, ys_mem_9325_ext.nbytes,
                                        "ys_mem_copy_9485")
        if (ys_mem_9325_ext.nbytes != 0):
          cl.enqueue_copy(self.queue, ys_mem_copy_9485, ys_mem_9325,
                          dest_offset=np.long(0), src_offset=np.long(0),
                          byte_count=np.long(ys_mem_9325_ext.nbytes))
        if synchronous:
          sync(self)
        time_start = time.time()
        (out_mem_9418,
         out_arrsizze_9419) = self.futhark_train_binary(xs_mem_copy_9484,
                                                        ys_mem_copy_9485,
                                                        n_8716, m_8717, n_8718)
        sync(self)
        time_end = time.time()
        if runtime_file:
          runtime_file.write(str((int((time_end * 1000000)) - int((time_start * 1000000)))))
          runtime_file.write("\n")
  except AssertionError as e:
    sys.exit("Assertion.{} failed".format(e))
  if runtime_file:
    runtime_file.close()
  write_value(cl.array.Array(self.queue, (out_arrsizze_9419,), ct.c_double, data=out_mem_9418).get(),
              binary=binary_output)
  sys.stdout.write("\n")
entry_points = {"train_binary": entry_train_binary}
entry_point_fun = entry_points.get(entry_point)
if (entry_point_fun == None):
  sys.exit("No entry point '{}'.  Select another with --entry point.  Options are:\n{}".format(entry_point,
                                                                                               "\n".join(entry_points.keys())))
else:
  entry_point_fun()