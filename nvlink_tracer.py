#!/usr/bin/python3
import argparse
import enum
import subprocess
import re
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class NvLinkCounters:
    tx_kib: int = 0
    rx_kib: int = 0

    def __sub__(self, other):
        return NvLinkCounters(
            tx_kib=self.tx_kib - other.tx_kib,
            rx_kib=self.rx_kib - other.rx_kib
        )

    def total_traffic_kib(self) -> int:
        return self.tx_kib + self.rx_kib

    def __bool__(self):
        return self.tx_kib > 0 or self.rx_kib > 0


@dataclass
class GpuNvLinkState:
    gpu_id: int
    gpu_uuid: str
    gpu_name: str
    links: Dict[int, NvLinkCounters]


def parse_nvidia_smi_nvlink(output: str, raw: bool) -> List[GpuNvLinkState]:
    """Parse output of: nvidia-smi nvlink -gt d"""
    lines = output.strip().splitlines()
    gpus = []
    current_gpu = None
    link_pattern = re.compile(r"Link (\d+): {} ([a-zA-Z]{{2}}): (\d+) KiB".format("Raw" if raw else "Data"))

    for line in lines:
        line = line.strip()
        if line.startswith("GPU "):
            match = re.match(r"GPU (\d+): (.+) \(UUID: GPU-([a-f0-9-]+)\)", line)
            if match:
                gpu_id = int(match.group(1))
                gpu_name = match.group(2)
                gpu_uuid = match.group(3)
                current_gpu = GpuNvLinkState(
                    gpu_id=gpu_id,
                    gpu_uuid=gpu_uuid,
                    gpu_name=gpu_name,
                    links={}
                )
                gpus.append(current_gpu)
            continue

        if current_gpu:
            m = link_pattern.match(line)
            if m:
                link_id = int(m.group(1))
                direction = m.group(2)
                value = int(m.group(3))
                link = current_gpu.links.setdefault(link_id, NvLinkCounters())
                if direction == "Tx":
                    link.tx_kib = value
                elif direction == "Rx":
                    link.rx_kib = value
    return gpus


def get_nvlink_state(raw: bool) -> List[GpuNvLinkState]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "-gt", "r" if raw else "d"],
            capture_output=True,
            text=True,
            check=True
        )
        return parse_nvidia_smi_nvlink(result.stdout, raw)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed: {e.stderr.strip()}") from e


def run_benchmark(cmd: List[str]) -> subprocess.CompletedProcess:
    if not cmd:
        raise ValueError("No command provided to benchmark.")
    return subprocess.run(cmd, capture_output=True, text=True)


def diff_nvlink_states(
    before: List[GpuNvLinkState],
    after: List[GpuNvLinkState]
) -> List[GpuNvLinkState]:
    if len(before) != len(after):
        raise ValueError("GPU count mismatch between before and after")
    diffed = []
    for b, a in zip(before, after):
        if b.gpu_uuid != a.gpu_uuid:
            raise ValueError(f"GPU UUID mismatch: {b.gpu_uuid} vs {a.gpu_uuid}")
        diff_links = {}
        for link_id in range(18):
            b_link = b.links.get(link_id, NvLinkCounters())
            a_link = a.links.get(link_id, NvLinkCounters())
            diff_links[link_id] = a_link - b_link
        diffed_gpu = GpuNvLinkState(
            gpu_id=b.gpu_id,
            gpu_uuid=b.gpu_uuid,
            gpu_name=b.gpu_name,
            links=diff_links
        )
        diffed.append(diffed_gpu)
    return diffed


class Dir(enum.Enum):
    TX = "TX"
    RX = "RX"


def find_active_nvlinks(diff_states: List[GpuNvLinkState], threshold_kib: int = 0) -> Dict[Tuple[int, Dir], List[int]]:
    active = {}
    for state in diff_states:
        active_rx = [
            link_id for link_id, counters in state.links.items()
            if counters.rx_kib > threshold_kib
        ]
        active_tx = [
            link_id for link_id, counters in state.links.items()
            if counters.tx_kib > threshold_kib
        ]
        if active_rx:
            active[(state.gpu_id, Dir.RX)] = active_rx
        if active_tx:
            active[(state.gpu_id, Dir.TX)] = active_tx

    return active


def print_active_nvlinks(active: Dict[Tuple[int, Dir], List[int]]):
    if not active:
        print("No NVLinks showed significant traffic.")
        return
    print("Active NVLinks (per GPU):")
    for (gpu_id, gpu_dir), links in sorted(active.items(), key=lambda x: x[0][0]):
        print(f" GPU{gpu_id} {gpu_dir.value}: Links {', '.join(map(str, sorted(links)))}")


def print_traffic_summary(diff_states: List[GpuNvLinkState]):
    print("\nNVLink Traffic Summary (Tx + Rx in KiB):")
    for state in diff_states:
        total = sum(c.total_traffic_kib() for c in state.links.values())
        active_count = sum(1 for c in state.links.values() if c)
        print(f" GPU {state.gpu_id} ({state.gpu_name.split()[1]}): "
              f"{total:,} KiB total, {active_count}/18 links active")


def to_json_serializable(diff_states: List[GpuNvLinkState], extra: Optional[str]) -> List[dict]:
    return [
        {
            "gpu_id": d.gpu_id,
            "gpu_uuid": d.gpu_uuid,
            "gpu_name": d.gpu_name,
            "extra": extra,
            "links": {
                str(link_id): {"tx_kib": c.tx_kib, "rx_kib": c.rx_kib}
                for link_id, c in d.links.items()
                if c  # only include links with traffic
            }
        }
        for d in diff_states
        if any(c for c in d.links.values())
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Measure NVLink traffic during execution of a command.",
        epilog="Example: ./nvlink_tracer.py --json -- sleep 1"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results in JSON format only"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Use raw nvlink metrics including protocol overhead"
    )
    parser.add_argument(
        "--silent", action="store_true",
        help="Does not show output of benchmark"
    )
    parser.add_argument(
        "--extra",
        help="Add extra in json"
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run"
    )

    args = parser.parse_args()
    
    cmd = args.command
    if not cmd:
        parser.error("No commands provided")

    try:
        before = get_nvlink_state(args.raw)
        result = run_benchmark(cmd)

        if not args.silent:
            print(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(f"Benchmark command failed (exit {result.returncode}):\n{result.stderr}")

        after = get_nvlink_state(args.raw)
        diff = diff_nvlink_states(before, after)
        active = find_active_nvlinks(diff, threshold_kib=0)

        if args.json:
            json_output = to_json_serializable(diff, args.extra)
            json.dump(json_output, sys.stdout, indent=2)
            print()
        else:
            print_active_nvlinks(active)
            print_traffic_summary(diff)

    except Exception as e:
        if args.json:
            json.dump({"error": str(e)}, sys.stdout, indent=2)
            print()
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
