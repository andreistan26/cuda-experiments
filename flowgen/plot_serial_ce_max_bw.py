import matplotlib.pyplot as plt
import json
import sys


DEFAULT_STATS = "results/serial-ce/stats.json"


def fmt_bytes(size):
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if value.is_integer():
        return f"{int(value)}{unit}"
    return f"{value:.1f}{unit}"


def main():
    if len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} [<flowgen-stats.json>] [<output-fig.png>]")
        exit(1)

    stats_path = DEFAULT_STATS
    if len(sys.argv) >= 2:
        stats_path = sys.argv[1]

    output_path = None
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    with open(stats_path, "r") as f:
        results = json.load(f)

    sm_flows = list(filter(lambda r: r["type"].startswith("SM"), sorted(results["flows"], key=lambda flow: flow["buffer_size"])))
    ce_flows = list(filter(lambda r: r["type"].startswith("CE"), sorted(results["flows"], key=lambda flow: flow["buffer_size"])))
    buffer_sizes = []
    max_bws_sm = []
    max_bws_ce = []
    labels = []
    for flow in sm_flows:
        iters = flow["iterations"]
        best_iter = max(iters, key=lambda run: run["bandwidth_GBs"])
        buffer_sizes.append(flow["buffer_size"])
        max_bws_sm.append(best_iter["bandwidth_GBs"])
        labels.append(fmt_bytes(flow["buffer_size"]))

    for flow in ce_flows:
        iters = flow["iterations"]
        best_iter = max(iters, key=lambda run: run["bandwidth_GBs"])
        max_bws_ce.append(best_iter["bandwidth_GBs"])

    _, ax = plt.subplots(figsize=(12, 8))
    ax.plot(buffer_sizes, max_bws_sm,
            marker="o",
            linewidth=3,
            label="Persistent SM Flow bandwidth")
    ax.plot(buffer_sizes, max_bws_ce,
            marker="^",
            linewidth=3,
            label="CE Flow bandwidth")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Buffer size", fontsize="x-large")
    ax.set_ylabel("Bandwidth(GB/s)", fontsize="x-large")
    ax.set_xticks(buffer_sizes)
    ax.set_xscale('log')
    ax.set_xticklabels(labels)
    ax.legend(fontsize="large")
    ax.grid()
    ax.grid(which="minor", color="0.9")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()
