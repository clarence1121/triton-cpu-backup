import os
import hashlib
import importlib
import importlib.resources
import tempfile
import time

import triton
import triton._C
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

from pathlib import Path
from triton._C.libtriton import llvm

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = []
library_dirs = [_triton_C_dir]
libraries = ["stdc++"]

# Skip non-existent paths
sys_include_dir = os.path.join(_dirname, "include")
if os.path.exists(sys_include_dir):
    include_dirs.append(sys_include_dir)

sys_lib_dir = os.path.join(_dirname, "lib")
if os.path.exists(sys_lib_dir):
    library_dirs.append(sys_lib_dir)


def compile_module_from_src(inc, src, kernel_name):
    launcher_include_dir = os.getenv("KERNEL_LAUNCHER_INCLUDE_DIR")
    launcher_src_dir = os.getenv("KERNEL_AUX_FILE_DIR")


    block_shape = os.getenv("TUNING_SHAPE_CONFIG")
    if block_shape is None:
        block_shape =""

    if launcher_include_dir is None:
       launcher_include_dir = tempfile.mkdtemp()

    if launcher_src_dir is None:
       launcher_src_dir = launcher_include_dir

    # launcher_include_dir +="/" + block_shape
    # launcher_src_dir +="/" + block_shape
    # launcher_include_dir += block_shape
    launcher_src_dir += block_shape
    os.makedirs(launcher_include_dir, mode=0o777, exist_ok=True)
    os.makedirs(launcher_src_dir, mode=0o777, exist_ok=True)


    # print("launcher include dir: ", launcher_include_dir)
    # print("launcher src dir: ", launcher_src_dir)
    inc_path = os.path.join(launcher_include_dir, kernel_name+"_launcher.h")
    with open(inc_path, "w") as f:
        f.write(inc)

    src_path = os.path.join(launcher_src_dir, kernel_name+"_launcher.cpp")
    with open(src_path, "w") as f:
        f.write(src)

    # key = hashlib.md5(src.encode("utf-8")).hexdigest()
    # cache = get_cache_manager(key)
    # cache_path = cache.get_file(f"{name}.so")
    # if cache_path is None:
        # with tempfile.TemporaryDirectory() as tmpdir:
            # tmpdir = tempfile.mkdtemp()
            # print(tmpdir)
            # src_path = os.path.join(tmpdir, "main.cpp")
            # with open(src_path, "w") as f:
            #     f.write(src)
            # so = _build(name, src_path, tmpdir, library_dirs, include_dirs, libraries)
            # with open(so, "rb") as f:
            #     cache_path = cache.put(f.read(), f"{name}.so", binary=True)
    # import importlib.util
    # spec = importlib.util.spec_from_file_location(name, cache_path)
    # mod = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mod)
    # return mod

# ------------------------
# Utils
# ------------------------


class CPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            import ctypes
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 0, 0)

    def get_device_properties(self, *args):
        return {"max_shared_mem": 0}


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    if ty == "constexpr":
        return "int32_t"  # constexpr values are treated as int32_t
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids, kernel_name, constexprs_arg_names):
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors.
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    kernel_fn_args = [i for i in signature.keys() if i not in constants]
    kernel_fn_args_list = ', '.join(f"arg{i}" for i in kernel_fn_args)
    kernel_fn_arg_types = ', '.join([f"{ty_to_cpp(signature[i])}" for i in kernel_fn_args] + ["uint32_t"] * 6)

    kernel_constants_declare = "".join(f"extern const int {kernel_name}_{arg_name};\n" for arg_id, arg_name in constexprs_arg_names.items() if isinstance(constants[arg_id], int) )
    kernel_constants_definition = "".join(f"const int {kernel_name}_{arg_name} = {constants[arg_id]};\n" for arg_id, arg_name in constexprs_arg_names.items() if isinstance(constants[arg_id], int))


    # print(kernel_constants_declare)
    # print(kernel_constants_definition)

    inc = f"""
#include <stdint.h>
#include <cstddef>

using {kernel_name}_kernel_ptr_t = void(*)({kernel_fn_arg_types});

extern "C"{{
 // Pointer type (=Memref) becomes int64_t + MemRef struct
 // FIXME: understand what this int64_t is used for.
 void({kernel_name})({kernel_fn_arg_types});
}}

{kernel_constants_declare}

void {kernel_name}_omp(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                        {kernel_name}_kernel_ptr_t kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''});
"""

    # generate glue code
    src = f"""
#include "{kernel_name}_launcher.h"
#include "support/omp.h"
#include "support/support.h"
#include <algorithm>
#include <optional>
#include <stdio.h>

{kernel_constants_definition}

void {kernel_name}_omp(uint32_t gridX, uint32_t gridY, uint32_t gridZ, {kernel_name}_kernel_ptr_t kernel_ptr {', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
   // TODO: Consider using omp collapse(3) clause for simplicity?
   auto all_grids = get_all_grids(gridX, gridY, gridZ);
   size_t N = gridX * gridY * gridZ;

   std::optional<int> max_threads = getIntEnv("TRITON_CPU_MAX_THREADS");
   if (max_threads.has_value())
     max_threads = std::max(1, std::min(max_threads.value(), omp_get_max_threads()));
   else
     max_threads = omp_get_max_threads();

   #pragma omp parallel for schedule(static) num_threads(max_threads.value())
   for (size_t i = 0; i < N; ++i) {{
     const auto [x, y, z] = all_grids[i];
     (*kernel_ptr)({kernel_fn_args_list + ', ' if len(kernel_fn_args) > 0 else ''} x, y, z, gridX, gridY, gridZ);
   }}
 }}
 """
    return inc, src


class CPULauncher(object):

    def __init__(self, src, metadata, name):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i

        constexprs_arg_names = {cst_key(key): key for key, value in constants.items()  if(cst_key(key) in  src.fn.constexprs)}

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        inc, src = make_launcher(constants, signature, ids, name, constexprs_arg_names)
        compile_module_from_src(inc, src, name)
        # self.launch = mod.launch

    def __call__(self, *args, **kwargs):
        # self.launch(*args, **kwargs)
        pass


class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()


class CPUDriver(DriverBase):

    def __init__(self):
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        super().__init__()

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu", self.get_current_device())

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        # Capability and warp size are zeros for CPU.
        # TODO: GPUTarget naming isn't obviously good.
        cpu_arch = llvm.get_cpu_tripple().split("-")[0]
        return GPUTarget("cpu", cpu_arch, 0)

    def get_device_interface(self):
        return CPUDeviceInterface()

    @staticmethod
    def is_active():
        return True

    def get_benchmarker(self):
        from triton.testing import do_bench

        def do_bench_cpu(*args, **kwargs):
            if not 'measure_time_with_hooks' in kwargs:
                kwargs['measure_time_with_hooks'] = True
            return do_bench(*args, **kwargs)

        return do_bench_cpu

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cpu')

    # TODO maybe CPU should do anything here
    def clear_cache(self, cache):
        cache.zero_()
