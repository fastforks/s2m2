"""Custom CUDA kernels of s2m2_kernels.cu for PyTorch (loader adapted from RAFT-Stereo exp 002 cuda_kernels.py):
shipped prebuilt as a fatbin (core/model/prebuilt/, sm_80/86/89/90/120 SASS + compute_80 PTX, so no nvcc is needed);
only if it is missing (kernel source edited) a per-arch cubin is compiled once with CUDA_HOME's nvcc into
~/.cache/s2m2-torch. Loaded through the driver API, launched on torch's current stream. Size-independent.
Rebuild the shipped fatbin after editing s2m2_kernels.cu:  python -m s2m2.core.model.kernels
"""

import ctypes
import hashlib
import os
import struct
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2m2_kernels.cu")
_PREBUILT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "s2m2-torch")


def _stream():
    """torch's current raw CUDA stream (torch.cuda.current_stream() costs ~3 us per call)."""
    return torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice())


# p: pointer (tensor or int address), i: int32, l: int64, f: float32
SIGNATURES = {
    "bias_act": "ppplii",
    "bias_act8": "ppplii",
    "gate_mix8": "pppppli",
    "ot_col_lse": "pppiiff",
    "ot_row_lse": "pppiiiff",
    "ot_final": "pppiiifppp",
    "ln_rows": "pplif",
    "up2x_nhwc8": "ppiiiilii",
    "gate_mix8s": "ppppplil",
    "pool2x_nhwc8": "ppiiii",
    "gn_stats": "pplii",
    "affine_nc8": "pppli" + "i",
    "convex_up": "pppiiii",
    "cv_lookup_otf": "pppppppiiifff",
    "attn_d384": "ppppllllllii" + "f",
}
_STRUCT = {"p": "P", "i": "i", "l": "q", "f": "f", "u": "I"}
_FMT = {}


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


def _nvcc(arch_args, out):
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    subprocess.run(
        [
            os.path.join(cuda_home, "bin", "nvcc"),
            *arch_args,
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{cuda_home}/include",
            "-o",
            out + ".tmp",
            _SRC,
        ],
        check=True,
    )
    os.replace(out + ".tmp", out)


def _src_tag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build_prebuilt(archs=(80, 86, 89, 90, 120)):
    """Build the shipped fatbin for the current kernel source (run after editing s2m2_kernels.cu)."""
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("s2m2_kernels_"):
            os.remove(os.path.join(_PREBUILT, f))
    out = os.path.join(_PREBUILT, f"s2m2_kernels_{_src_tag()}.fatbin")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + [
        "-gencode=arch=compute_80,code=compute_80"
    ]
    _nvcc(["-fatbin", *gencode], out)
    return out


class Kernels:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        src = open(_SRC, "rb").read()
        # shipped fatbin (sm_80..sm_120 SASS + compute_80 PTX), else a per-arch cubin compiled once with nvcc
        cubin = os.path.join(_PREBUILT, f"s2m2_kernels_{_src_tag()}.fatbin")
        if not os.path.exists(cubin):
            tag = hashlib.sha1(src + f"sm_{major}{minor}".encode()).hexdigest()[:16]
            cubin = os.path.join(_CACHE, f"s2m2_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            os.makedirs(_CACHE, exist_ok=True)
            _nvcc(["-cubin", f"-arch=sm_{major}{minor}"], cubin)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint] * 7
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.module = ctypes.c_void_p()
        self._image = open(cubin, "rb").read()
        _check(
            self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData"
        )
        self.funcs = {}
        self._smem = {}
        for name in SIGNATURES:
            f = ctypes.c_void_p()
            _check(
                self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()),
                f"cuModuleGetFunction {name}",
            )
            self.funcs[name] = f

    @staticmethod
    def linear(n):
        """Rust's Launch::linear: 256 threads, grid-stride loops, at most 65535*8 blocks."""
        return (min(max((n + 255) // 256, 1), 65535 * 8), 1, 1), (256, 1, 1)

    def launch(self, name, cfg, *args, smem=0):
        """Arguments packed into one C-layout buffer (CU_LAUNCH_PARAM_BUFFER_POINTER): ~3x less Python overhead than
        one ctypes object per argument."""
        grid, block = cfg
        fmt = _FMT.get(name)
        if fmt is None:
            fmt = _FMT[name] = "@" + "".join(_STRUCT[t] for t in SIGNATURES[name])
        vals = [
            (a.data_ptr() if isinstance(a, torch.Tensor) else int(a or 0)) if t == "p" else a
            for t, a in zip(SIGNATURES[name], args)
        ]
        buf = struct.pack(fmt, *vals)
        size = ctypes.c_size_t(len(buf))
        extra = (ctypes.c_void_p * 5)(
            1,
            ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p).value,
            2,
            ctypes.addressof(size),
            0,
        )
        if (
            smem > 48 * 1024 and self._smem.get(name, 0) < smem
        ):  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _check(
                self.lib.cuFuncSetAttribute(self.funcs[name], 8, ctypes.c_int(smem)),
                "cuFuncSetAttribute",
            )
            self._smem[name] = smem
        _check(
            self.lib.cuLaunchKernel(
                self.funcs[name],
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                smem,
                _stream(),
                None,
                extra,
            ),
            f"launch {name}",
        )


if __name__ == "__main__":
    print(build_prebuilt())
