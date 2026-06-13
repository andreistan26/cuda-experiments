import matplotlib.pyplot as plt
import json
import sys

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <flowgen-stats.json> [<output-fig.png>]")
        exit(1)
    results = ""
    with open(sys.argv[1], "r") as f:
        results = json.load(f)
    print(results)
    flows = results["flows"]
    _, ax = plt.subplots()
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
        ax.set_yscale("log")
        ax.plot(flow_ts, flow_bw,
                 linewidth=3,
                 label=f"{src}->{dst} bs={buffer_size} ({driver})")
        ax.set_xlabel("Time(ms)", fontsize="x-large")
        ax.set_ylabel("Bandwidth(GB/s)", fontsize="x-large")
    ax.legend(fontsize="large")
    ax.grid()
    ax.grid(which="minor", color="0.9")
    if len(sys.argv) >= 3: plt.savefig(sys.argv[2])
    plt.show()

if __name__ == "__main__":
    main()
