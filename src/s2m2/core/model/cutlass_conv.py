"""Fused-epilogue convolutions / linears of s2m2_cutlass.cu (CUTLASS implicit GEMM, fp16 accumulation; origin: exp 003
IGEV-general cutlass_conv.py <- RAFT-Stereo exp 002). Shipped prebuilt (prebuilt/libs2m2_cutlass_<hash>.so, sm_80/86/89/90/120
SASS + compute_80 PTX); only without it (source edited) built once with CUDA_HOME's nvcc into ~/.cache (needs the CUTLASS
v4.7.1 headers: CUTLASS_DIR). The problem size is a runtime argument:
nothing is compiled or tuned per input size.  Rebuild the shipped library:  python -m s2m2.core.model.cutlass_conv
"""

import ctypes
import hashlib
import os
import subprocess

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "s2m2_cutlass.cu")
_PREBUILT = os.path.join(_HERE, "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "s2m2-torch")


def _stream():
    """torch's current raw CUDA stream (torch.cuda.current_stream() costs ~3 us per call)."""
    return torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice())


def _cutlass_dir():
    directory = os.environ.get("CUTLASS_DIR", "")
    if not os.path.isfile(os.path.join(directory, "include", "cutlass", "cutlass.h")):
        raise FileNotFoundError(
            "CUTLASS headers not found: set CUTLASS_DIR to a CUTLASS v4.7.1 checkout"
        )
    return directory


class EpiArgs(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("act", ctypes.c_int),
        ("b", ctypes.c_void_p),
        ("r", ctypes.c_void_p),
        ("rs", ctypes.c_longlong),
        ("y", ctypes.c_void_p),
        ("ys", ctypes.c_longlong),
        ("y2", ctypes.c_void_p),
        ("y2s", ctypes.c_longlong),
        ("h", ctypes.c_void_p),
        ("hs", ctypes.c_longlong),
        ("z", ctypes.c_void_p),
        ("zs", ctypes.c_longlong),
        ("split", ctypes.c_int),
    ]


def _tag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build(out, archs=(80, 86, 89, 90, 120)):
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + [
        "-gencode=arch=compute_80,code=compute_80"
    ]
    # nvcc.profile embeds its absolute library-search paths in the fatbin. Supply the toolkit explicitly.
    build_env = dict(
        os.environ,
        PATH=os.path.join(cuda_home, "nvvm", "bin") + os.pathsep + os.environ.get("PATH", ""),
    )
    subprocess.run(
        [
            os.path.join(cuda_home, "bin", "nvcc"),
            "--dont-use-profile",
            "-ldir",
            os.path.join(cuda_home, "nvvm", "libdevice"),
            f"-I{cuda_home}/include",
            "-isystem",
            os.path.join(cuda_home, "include", "cccl"),
            "-cudadevrt",
            "none",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-Xcompiler",
            f"-ffile-prefix-map={_cutlass_dir()}=cutlass",
            "-Xcompiler",
            f"-ffile-prefix-map={_HERE}=.",
            "-cudart",
            "shared",
            "-Xlinker",
            f"-L{cuda_home}/lib",
            *gencode,
            f"-I{_cutlass_dir()}/include",
            "-o",
            out + ".tmp",
            _SRC,
        ],
        check=True,
        env=build_env,
    )
    os.replace(out + ".tmp", out)
    return out


def build_prebuilt():
    os.makedirs(_PREBUILT, exist_ok=True)
    out = build(os.path.join(_PREBUILT, f"libs2m2_cutlass_{_tag()}.so"))
    for f in os.listdir(_PREBUILT):
        if f.startswith("libs2m2_cutlass_") and os.path.join(_PREBUILT, f) != out:
            os.remove(os.path.join(_PREBUILT, f))
    return out


def _p(t):
    return 0 if t is None else t.data_ptr()


class CutlassConvs:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()  # torch has loaded libcudart.so.13, which the library links against
        tag = _tag()
        lib = os.path.join(_PREBUILT, f"libs2m2_cutlass_{tag}.so")
        if not os.path.exists(lib):
            major, minor = torch.cuda.get_device_capability()
            lib = os.path.join(_CACHE, f"libs2m2_cutlass_{tag}_sm{major}{minor}.so")
            if not os.path.exists(lib):
                os.makedirs(_CACHE, exist_ok=True)
                build(lib, archs=(major * 10 + minor,))
        self.lib = ctypes.CDLL(lib)
        self.lib.s2m2_conv.argtypes = (
            [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            + [ctypes.c_int] * 10
            + [ctypes.POINTER(EpiArgs), ctypes.c_void_p]
        )

    def conv(self, cfg, x, w, padding, stride, e):
        """x: [N, C, H, W] channels-last fp16 (dense), w: [K, C, R, S] channels-last fp16. Returns the status
        (0 ok, 1 geometry not supported, else error)."""
        Nb, C, H, W = x.shape
        K, _, R, S = w.shape
        return self.lib.s2m2_conv(
            cfg,
            x.data_ptr(),
            w.data_ptr(),
            Nb,
            H,
            W,
            C,
            K,
            R,
            S,
            padding[0],
            padding[1],
            stride,
            ctypes.byref(e),
            _stream(),
        )


def epi_args(
    mode=0,
    act=0,
    b=None,
    r=None,
    rs=0,
    y=None,
    ys=0,
    y2=None,
    y2s=0,
    h=None,
    hs=0,
    z=None,
    zs=0,
    split=0,
):
    return EpiArgs(mode, act, _p(b), _p(r), rs, _p(y), ys, _p(y2), y2s, _p(h), hs, _p(z), zs, split)


if __name__ == "__main__":
    print(build_prebuilt())
