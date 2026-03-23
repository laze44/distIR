# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""GEMM tile mapping search across multiple devices.

Enumerates all valid tiling/mapping strategies for A*B=C across 16 fully-
connected devices and selects the one that minimises
    total_time = max(compute_time, communication_time)
for each M size.

Hardware parameters are read from distIR/config/nmp.json.

Usage:
    python gemm_tile_search.py --K 4096 --N 4096
    python gemm_tile_search.py --K 4096 --N 4096 --num_devices 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------

@dataclass
class HWConfig:
    """Hardware parameters for a single device."""
    peak_tflops_fp8: float       # TFLOPS (fp8)
    mem_bw_tb_s: float           # TB/s
    mem_capacity_gb: float       # GB
    inter_bw_gb_s: float         # GB/s  (inter-node bandwidth used for device link)
    inter_latency_us: float      # µs

    @classmethod
    def from_json(cls, path: str) -> "HWConfig":
        with open(path) as f:
            cfg = json.load(f)
        return cls(
            peak_tflops_fp8=cfg["compute"]["peak_tflops"]["fp8"],
            mem_bw_tb_s=cfg["memory"]["bandwidth_tb_per_s"],
            mem_capacity_gb=cfg["memory"]["capacity_gb"],
            inter_bw_gb_s=cfg["interconnect"]["inter_node"]["bandwidth_gb_per_s"],
            inter_latency_us=cfg["interconnect"]["inter_node"]["latency_us"],
        )


# ---------------------------------------------------------------------------
# Helper: factorizations
# ---------------------------------------------------------------------------

def factorizations_3d(n: int) -> List[Tuple[int, int, int]]:
    """Return all ordered (dm, dk, dn) with dm*dk*dn == n, dm/dk/dn >= 1."""
    results = []
    for dm in range(1, n + 1):
        if n % dm != 0:
            continue
        rem = n // dm
        for dk in range(1, rem + 1):
            if rem % dk != 0:
                continue
            dn = rem // dk
            results.append((dm, dk, dn))
    return results


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class MappingResult:
    """Stores the cost analysis for a single (dm, dk, dn) mapping."""
    dm: int
    dk: int
    dn: int
    M: int
    K: int
    N: int
    # per-device tile sizes
    tile_m: int
    tile_k: int
    tile_n: int
    # roofline model per device
    flops_per_device: float       # FLOPs
    mem_bytes_per_device: float   # bytes loaded/stored from device memory
    arith_intensity: float        # FLOPs / byte  (operational intensity)
    ridge_point: float            # peak_flops / mem_bw  (FLOP/byte)
    compute_time_us: float        # max(flops/peak, mem_bytes/mem_bw)
    roofline_regime: str          # "compute" or "memory"
    # communication
    k_reduce_bytes: float         # bytes per device for K-reduction (SHARP reduce)
    comm_time_us: float
    # total
    total_time_us: float


def evaluate_mapping(
    M: int, K: int, N: int,
    dm: int, dk: int, dn: int,
    hw: HWConfig,
    num_devices: int,
    elem_bytes: float = 1.0,     # fp8 = 1 byte
) -> MappingResult | None:
    """Evaluate a (dm, dk, dn) mapping.  Returns None if infeasible."""

    # Tile sizes (must be integer-divisible for simplicity; skip if not)
    if M % dm != 0 or K % dk != 0 or N % dn != 0:
        return None

    tile_m = M // dm
    tile_k = K // dk
    tile_n = N // dn

    # --- Compute cost (per device) – Roofline model ---
    # Each device computes tile_m x tile_k @ tile_k x tile_n
    # FLOPs = 2 * tile_m * tile_k * tile_n  (multiply-add)
    flops = 2.0 * tile_m * tile_k * tile_n

    # Memory traffic: read A tile + read B tile + write C tile
    a_tile_bytes = tile_m * tile_k * elem_bytes
    b_tile_bytes = tile_k * tile_n * elem_bytes
    c_tile_bytes_local = tile_m * tile_n * elem_bytes
    mem_bytes = a_tile_bytes + b_tile_bytes + c_tile_bytes_local

    # Arithmetic intensity (FLOP/Byte)
    arith_intensity = flops / mem_bytes if mem_bytes > 0 else float('inf')

    # Hardware ceilings
    peak_flops_per_s = hw.peak_tflops_fp8 * 1e12      # TFLOPS → FLOPS
    mem_bw_bytes_per_s = hw.mem_bw_tb_s * 1e12         # TB/s  → B/s
    ridge_point = peak_flops_per_s / mem_bw_bytes_per_s  # FLOP/Byte

    # Roofline: attainable perf = min(peak, AI * mem_bw)
    #   compute_time = flops / attainable_perf
    #                = max(flops / peak, mem_bytes / mem_bw)
    compute_bound_time_s = flops / peak_flops_per_s
    memory_bound_time_s  = mem_bytes / mem_bw_bytes_per_s
    compute_time_s = max(compute_bound_time_s, memory_bound_time_s)
    compute_time_us = compute_time_s * 1e6

    if arith_intensity >= ridge_point:
        roofline_regime = "compute"
    else:
        roofline_regime = "memory"

    # --- Communication cost ---
    inter_bw_bytes_per_s = hw.inter_bw_gb_s * 1e9      # GB/s → B/s

    # B multicast: NOT needed.
    # A matrix can be freely replicated across devices (pre-loaded), so every
    # device already holds the A tile it needs.  B is uniquely partitioned
    # (no replication) and each device owns exactly one B tile which it uses
    # locally.  Therefore there is zero B communication cost.

    # K-reduction with SHARP (in-network computing)
    #   When dk > 1, each C[i_m, i_n] block has dk partial products that must
    #   be summed.  With NVSwitch SHARP the network switch performs the
    #   reduction in-flight:
    #     - Each device sends its partial C tile to the switch   → C_tile_bytes
    #     - The switch reduces all dk contributions and sends
    #       the result back to one (or all) devices              → C_tile_bytes
    #   Per-device volume = 2 * C_tile_bytes  (one send + one receive),
    #   independent of dk.
    #   Bottleneck time = C_tile_bytes / bw  (send and receive overlap on
    #   full-duplex links, so the bottleneck is one direction).
    c_tile_bytes = tile_m * tile_n * elem_bytes
    if dk > 1:
        k_reduce_bytes_per_device = 2.0 * c_tile_bytes   # send + receive
        # Full-duplex: send and receive overlap → bottleneck = max(send, recv)
        k_reduce_time_s = c_tile_bytes / inter_bw_bytes_per_s
    else:
        k_reduce_bytes_per_device = 0.0
        k_reduce_time_s = 0.0

    # Total communication time (only K-reduction)
    comm_time_s = k_reduce_time_s
    comm_time_us = comm_time_s * 1e6

    # --- Total time = max(compute, communication) ---
    total_time_us = max(compute_time_us, comm_time_us)

    return MappingResult(
        dm=dm, dk=dk, dn=dn,
        M=M, K=K, N=N,
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n,
        flops_per_device=flops,
        mem_bytes_per_device=mem_bytes,
        arith_intensity=arith_intensity,
        ridge_point=ridge_point,
        compute_time_us=compute_time_us,
        roofline_regime=roofline_regime,
        k_reduce_bytes=k_reduce_bytes_per_device,
        comm_time_us=comm_time_us,
        total_time_us=total_time_us,
    )


# ---------------------------------------------------------------------------
# Search routine
# ---------------------------------------------------------------------------

def search_best_mapping(
    M: int, K: int, N: int,
    hw: HWConfig,
    num_devices: int = 16,
    elem_bytes: float = 1.0,
) -> Tuple[MappingResult | None, List[MappingResult]]:
    """Search all (dm, dk, dn) mappings and return (best, all_valid)."""
    factorizations = factorizations_3d(num_devices)
    results: List[MappingResult] = []

    for dm, dk, dn in factorizations:
        r = evaluate_mapping(M, K, N, dm, dk, dn, hw, num_devices, elem_bytes)
        if r is not None:
            results.append(r)

    if not results:
        return None, results

    best = min(results, key=lambda r: r.total_time_us)
    return best, results


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(b: float) -> str:
    if b < 1024:
        return f"{b:.0f} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"


def _fmt_flops(f: float) -> str:
    if f < 1e6:
        return f"{f:.0f}"
    if f < 1e9:
        return f"{f / 1e6:.2f} MFLOP"
    if f < 1e12:
        return f"{f / 1e9:.2f} GFLOP"
    return f"{f / 1e12:.4f} TFLOP"


def print_summary_table(
    K: int, N: int,
    hw: HWConfig,
    num_devices: int = 16,
    elem_bytes: float = 1.0,
) -> None:
    """Print a summary table for all M sizes (powers of 2 from 1 to 1024)."""

    m_values = [2 ** i for i in range(11)]  # 1, 2, 4, ..., 1024

    sep = "=" * 150
    print(sep)
    print(f"GEMM Tile Mapping Search  |  K={K}, N={N}  |  {num_devices} devices  |  fp8 ({elem_bytes}B)")
    print(f"Device: {hw.peak_tflops_fp8} TFLOPS(fp8), "
          f"MemBW={hw.mem_bw_tb_s} TB/s, "
          f"InterBW={hw.inter_bw_gb_s} GB/s  (SHARP in-network reduce)")
    ridge = hw.peak_tflops_fp8 * 1e12 / (hw.mem_bw_tb_s * 1e12)
    print(f"Roofline ridge point: {ridge:.1f} FLOP/Byte")
    print(sep)
    header = (
        f"{'M':>6s} | {'Best(dm,dk,dn)':>16s} | "
        f"{'TileM':>6s} {'TileK':>6s} {'TileN':>6s} | "
        f"{'FLOPs/dev':>14s} {'MemB/dev':>10s} {'AI':>8s} | "
        f"{'Comp(µs)':>12s} {'Regime':>8s} | "
        f"{'K_reduce':>10s} {'Comm(µs)':>12s} | "
        f"{'Total(µs)':>12s}"
    )
    print(header)
    print("-" * 150)

    for M in m_values:
        best, all_results = search_best_mapping(M, K, N, hw, num_devices, elem_bytes)
        if best is None:
            print(f"{M:>6d} | {'NO VALID MAPPING':>16s} |")
            continue

        mapping_str = f"({best.dm},{best.dk},{best.dn})"
        print(
            f"{M:>6d} | {mapping_str:>16s} | "
            f"{best.tile_m:>6d} {best.tile_k:>6d} {best.tile_n:>6d} | "
            f"{_fmt_flops(best.flops_per_device):>14s} {_fmt_bytes(best.mem_bytes_per_device):>10s} "
            f"{best.arith_intensity:>8.1f} | "
            f"{best.compute_time_us:>12.4f} {best.roofline_regime:>8s} | "
            f"{_fmt_bytes(best.k_reduce_bytes):>10s} {best.comm_time_us:>12.4f} | "
            f"{best.total_time_us:>12.4f}"
        )

    print(sep)


def print_all_mappings(
    M: int, K: int, N: int,
    hw: HWConfig,
    num_devices: int = 16,
    elem_bytes: float = 1.0,
) -> None:
    """Print all valid mappings for a specific M, sorted by total_time."""

    best, all_results = search_best_mapping(M, K, N, hw, num_devices, elem_bytes)
    if not all_results:
        print(f"No valid mappings for M={M}, K={K}, N={N}")
        return

    all_results.sort(key=lambda r: r.total_time_us)

    ridge = hw.peak_tflops_fp8 * 1e12 / (hw.mem_bw_tb_s * 1e12)
    sep = "=" * 170
    print(sep)
    print(f"All mappings for M={M}, K={K}, N={N}  |  {num_devices} devices  |  fp8 ({elem_bytes}B)  "
          f"|  SHARP reduce  |  ridge={ridge:.1f} FLOP/B")
    print(sep)
    header = (
        f"{'#':>3s} | {'(dm,dk,dn)':>12s} | "
        f"{'TileM':>6s} {'TileK':>6s} {'TileN':>6s} | "
        f"{'FLOPs/dev':>14s} {'MemB/dev':>10s} {'AI':>8s} | "
        f"{'Comp(µs)':>12s} {'Regime':>8s} | "
        f"{'K_reduce':>10s} {'Comm(µs)':>12s} | "
        f"{'Total(µs)':>12s} | {'Bottleneck':>10s}"
    )
    print(header)
    print("-" * 170)

    for idx, r in enumerate(all_results):
        mapping_str = f"({r.dm},{r.dk},{r.dn})"
        bottleneck = "compute" if r.compute_time_us >= r.comm_time_us else "comm"
        marker = " *" if r is best else ""
        print(
            f"{idx+1:>3d} | {mapping_str:>12s} | "
            f"{r.tile_m:>6d} {r.tile_k:>6d} {r.tile_n:>6d} | "
            f"{_fmt_flops(r.flops_per_device):>14s} {_fmt_bytes(r.mem_bytes_per_device):>10s} "
            f"{r.arith_intensity:>8.1f} | "
            f"{r.compute_time_us:>12.4f} {r.roofline_regime:>8s} | "
            f"{_fmt_bytes(r.k_reduce_bytes):>10s} {r.comm_time_us:>12.4f} | "
            f"{r.total_time_us:>12.4f} | {bottleneck:>10s}{marker}"
        )

    print(sep)
    print(f"Best mapping: ({best.dm},{best.dk},{best.dn})  total_time={best.total_time_us:.4f} µs")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Search optimal GEMM tile mapping across multiple devices."
    )
    parser.add_argument("--K", type=int, default=4096, help="K dimension of A(MxK) and B(KxN)")
    parser.add_argument("--N", type=int, default=4096, help="N dimension of B(KxN) and C(MxN)")
    parser.add_argument("--num_devices", type=int, default=16, help="Number of devices")
    parser.add_argument("--elem_bytes", type=float, default=1.0,
                        help="Element size in bytes (fp8=1, fp16=2, fp32=4)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to hardware config JSON (default: distIR/config/nmp.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all mappings for each M size")
    parser.add_argument("--M", type=int, default=None,
                        help="If set, only evaluate this specific M value (not the sweep)")
    args = parser.parse_args()

    # Locate config
    if args.config:
        config_path = args.config
    else:
        # Try relative to this file first, then relative to cwd
        this_dir = Path(__file__).resolve().parent
        config_path = this_dir.parent / "config" / "nmp.json"
        if not config_path.exists():
            config_path = Path("distIR/config/nmp.json")

    hw = HWConfig.from_json(str(config_path))
    print(f"Loaded HW config from: {config_path}")
    print(f"  peak_tflops(fp8)={hw.peak_tflops_fp8}, mem_bw={hw.mem_bw_tb_s} TB/s, "
          f"inter_bw={hw.inter_bw_gb_s} GB/s, inter_lat={hw.inter_latency_us} µs")
    print()

    if args.M is not None:
        # Single M evaluation: show all mappings
        print_all_mappings(args.M, args.K, args.N, hw, args.num_devices, args.elem_bytes)
    else:
        # Sweep M from 1 to 1024 (powers of 2)
        print_summary_table(args.K, args.N, hw, args.num_devices, args.elem_bytes)

        if args.verbose:
            print("\n\nDetailed breakdown per M:\n")
            m_values = [2 ** i for i in range(11)]
            for M in m_values:
                print_all_mappings(M, args.K, args.N, hw, args.num_devices, args.elem_bytes)


if __name__ == "__main__":
    main()
