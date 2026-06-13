# Flowgen

Synthetic traffic generator for scale-up.

- `./tests` contains the input presets
- `./plot_report_bw.py` plotting the results from `--output-json`

There are multiple types of `algos` supported for actually transfering the data:

- SM_SIMPLE -> simple kernel no unrolling
- SM_UNROLL -> faster kernel with unrolling imported from `nvbandwidth`
- SM_SPACED -> long lived kernel that executes all `--iters` in a single kernel and allows for spaced writes within the same kernel, recommended when doing spaced writes with SM-based copies due to amortizing the kernel launch overhead
- CE -> Copy engine

As `spaced` 

## Usage

10 iterations and 1 warmup of copy engine transfers from GPU 0 -> GPU 1
```bash
./flowgen --pattern single --algo CE --size 1G --w 1 --iters 10
```

10 iterations and 1 warmup of SM based transfers in all to all pattern
```bash
./flowgen --pattern alltoall --algo SM_UNROLL --size 512m --w 1 --iters 10
```

json input config and json results output
```bash
./flowgen --config-json ./tests/big_small.json --output-json > out-stats.json
python3 plot_report_bw.py out-stats.json
```

Perfetto trace output
```bash
./flowgen --config-json ./tests/big_small.json --output-perfetto perfetto-trace.json
```

> The trace can be visualized at https://ui.perfetto.dev/#!/explore
