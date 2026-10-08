"""Holds GPU memory so a larger card behaves like a smaller one (e.g. a 16 GB RTX 5070 Ti
like a 12 GB card) while Kokon runs. Uses only the CUDA runtime DLL that ships
with stable-diffusion.cpp (no Python packages).

    python scripts\\dev\\vram-ballast.py --card-mib 12282          # emulate a 12 GB card
    python scripts\\dev\\vram-ballast.py --mib 4000 --seconds 5    # quick self-test

--card-mib is the total a real card reports (12 GB cards: 12282 MiB in nvidia-smi); the
ballast is this card's total minus that. Desktop and other apps keep using what they use, as
they would on the smaller card. The memory is written every half second so Windows keeps it
resident instead of paging it out to shared memory. Ctrl+C frees it.

Limits of the emulation: memory bandwidth and compute stay those of the bigger card, so
times are faster than on a real 12 GB card (an RTX 3060 has ~40 % of a 5070 Ti's bandwidth).
"""

import argparse
import ctypes
import os
import sys
import time

MIB = 1024 * 1024
CHUNK_MIB = 256


def load_cudart(sdcpp_dir):
    os.add_dll_directory(sdcpp_dir)
    try:
        return ctypes.CDLL(os.path.join(sdcpp_dir, "cudart64_12.dll"))
    except OSError as e:
        sys.exit(f"cudart64_12.dll not found in {sdcpp_dir} ({e}). Pass --sdcpp <stable-diffusion.cpp folder>.")


def check(rt, code, what):
    if code != 0:
        rt.cudaGetErrorString.restype = ctypes.c_char_p
        sys.exit(f"{what} failed: {rt.cudaGetErrorString(code).decode()} ({code})")


def mem_info(rt):
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    check(rt, rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
    return free.value // MIB, total.value // MIB


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--card-mib", type=int, help="total VRAM of the card to emulate, as nvidia-smi reports it (12 GB: 12282)")
    g.add_argument("--mib", type=int, help="hold exactly this many MiB")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0, help="release after this long (0 = until Ctrl+C)")
    ap.add_argument("--sdcpp", default=r"C:\AI\sdcpp", help="folder with cudart64_12.dll")
    a = ap.parse_args()

    rt = load_cudart(a.sdcpp)
    rt.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    check(rt, rt.cudaSetDevice(a.device), "cudaSetDevice")
    free, total = mem_info(rt)
    want = a.mib if a.mib is not None else total - a.card_mib
    if want <= 0:
        sys.exit(f"Nothing to hold: the card has {total} MiB.")
    if want > free - 256:
        sys.exit(f"Only {free} MiB free of {total}; cannot hold {want} MiB. Close the app (stop sd-server) first.")

    chunks = []
    left = want
    while left > 0:
        n = min(CHUNK_MIB, left) * MIB
        p = ctypes.c_void_p()
        check(rt, rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)), "cudaMalloc")
        chunks.append((p, n))
        left -= n // MIB
    free_after, _ = mem_info(rt)
    print(f"Holding {want} MiB on GPU {a.device} ({total} MiB card): {free_after} MiB free now "
          f"(like a {total - want} MiB card). Ctrl+C to release.", flush=True)

    start = time.time()
    last_report = 0.0
    try:
        while a.seconds <= 0 or time.time() - start < a.seconds:
            for p, n in chunks:
                check(rt, rt.cudaMemset(p, 0, ctypes.c_size_t(n)), "cudaMemset")
            check(rt, rt.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
            if time.time() - last_report > 30:
                last_report = time.time()
                f, _ = mem_info(rt)
                print(time.strftime("%H:%M:%S"), f"free {f} MiB", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    for p, _ in chunks:
        rt.cudaFree(p)
    print("Released.")


if __name__ == "__main__":
    main()
