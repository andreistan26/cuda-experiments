import matplotlib.pyplot as plt
import json
import sys

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
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <flowgen-stats.json> [<output-fig.png>]")
        exit(1)
    results = ""
    with open(sys.argv[1], "r") as f:
        results = json.load(f)
    print(results)
    flows = results["flows"]
    _, ax = plt.subplots(figsize=(12, 8))
    for flow in flows:
        src = flow["src"]
        dst = flow["dst"]
        buffer_size = flow["buffer_size"]
        driver = flow["type"]
        if driver.find("SM") != -1: driver = "SM"
        runs = flow["iterations"]
        print(flow)
        flow_ts = []
        flow_bw = []
        for run in runs:
            flow_ts.append(run["launch_us"])
            flow_bw.append(run["bandwidth_GBs"])
            flow_ts.append(run["launch_us"] + run["latency_us"])
            flow_bw.append(run["bandwidth_GBs"])
            # NaN breaks the line between consecutive runs
            flow_ts.append(float("nan"))
            flow_bw.append(float("nan"))
        #ax.set_yscale("log")
        ax.plot(flow_ts, flow_bw,
                 linewidth=3,
                 label=f"{src}->{dst} bs={fmt_bytes(buffer_size)} ({driver})")
        ax.set_xlabel("Time(us)", fontsize="x-large")
        ax.set_ylabel("Bandwidth(GB/s)", fontsize="x-large")
    legend = ax.legend(fontsize="medium", loc='upper right', bbox_to_anchor=(1,1))
    ax.grid()
    _, right = ax.get_xlim()
    ax.set_xlim(right=(right * 1.2))
    ax.grid(which="minor", color="0.9")
    if len(sys.argv) >= 3: plt.savefig(sys.argv[2])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


    # 11:30 16:00
