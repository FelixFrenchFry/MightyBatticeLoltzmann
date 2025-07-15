#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import matplotlib
matplotlib.use('Agg') # fast backend
import matplotlib.pyplot as plt



# performance based on cell count
# SOURCE: cuda_mpi_04/benchmarks_steps_single_fp32.txt
perf_data_single_fp32 = {
    'H100':    [(100, 0.5), (200, 2.0), (300, 4.5), (400, 7.6), (500, 10.8), (1000, 18.7), (1500, 22.6), (2000, 24.4),
                (2500, 25.8), (3000, 26.5), (3500, 26.8), (5000, 28.0), (7000, 28.7), (9000, 28.7), (11000, 28.7), (13000, 28.2),
                (15000, 28.7), (20000, 28.8), (25000, 28.7), (30000, 28.6)],
    'A100':    [(100, 0.4), (200, 1.8), (300, 3.6), (400, 6.1), (500, 8.4), (1000, 13.7), (1500, 17.5), (2000, 18.5),
                (2500, 19.3), (3000, 19.6), (3500, 19.7), (5000, 20.2), (7000, 20.5), (9000, 20.6), (11000, 20.6), (13000, 20.2),
                (15000, 20.4), (20000, 21.0), (25000, 20.7), (30000, 20.6)],
    '5070 Ti': [(100, 0.1), (200, 0.7), (300, 1.3), (400, 2.0), (500, 3.7), (1000, 6.6), (1500, 7.9), (2000, 8.6),
                (2500, 9.0), (3000, 9.2), (3500, 9.5), (5000, 9.9), (7000, 10.1), (9000, 10.0), (11000, 10.3), (13000, 9.4)]
}

# performance based on cell count
# SOURCE: cuda_mpi_04/benchmarks_steps_single_fp64.txt
perf_data_single_fp64 = {
    'H100':    [(100, 0.4), (200, 1.9), (300, 4.1), (400, 6.7), (500, 9.0), (1000, 11.1), (1500, 12.7), (2000, 13.4),
                (2500, 13.7), (3000, 14.0), (3500, 14.0), (5000, 14.4), (7000, 14.4), (9000, 14.4), (11000, 14.5), (13000, 14.4),
                (15000, 14.5), (20000, 14.5)],
    'A100':    [(100, 0.4), (200, 1.7), (300, 3.4), (400, 5.5), (500, 6.6), (1000, 8.5), (1500, 9.7), (2000, 10.3),
                (2500, 10.4), (3000, 10.6), (3500, 10.7), (5000, 10.7), (7000, 10.7), (9000, 10.7), (11000, 10.7), (13000, 10.9),
                (15000, 10.8), (20000, 10.9)],
    '5070 Ti': [(100, 0.1), (200, 0.4), (300, 0.9), (400, 1.2), (500, 1.4), (1000, 2.0), (1500, 2.3), (2000, 2.4),
                (2500, 2.4), (3000, 2.4), (3500, 2.5), (5000, 2.5), (7000, 2.4), (9000, 2.4)]
}

# style
font_size = 18
line_width = 2.5
marker_size = 10
colors = {
    'H100': '#ff0032',
    'A100': '#2846b4',
    '5070 Ti': '#28b450',
}
fp64_style = {
    'linestyle': 'dotted',
    'alpha': 1.0
}

# mode selection
mode = "fp32"
plot_fp64 = True

outputPath = "output/bench/bench_single_fp32_fp64.png" if plot_fp64 else (
    "output/bench/bench_single_fp32.png" if mode == "fp32" else "output/bench/bench_single_fp64.png"
)

# plotting
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': font_size})

for gpu_model, data_fp32 in perf_data_single_fp32.items():

    # skip placeholder data
    if not data_fp32 or all((x == 0 and y == 0) for x, y in data_fp32):
        continue

    x_dim_fp32, blups_fp32 = zip(*data_fp32)
    num_cells_fp32 = [x**2 for x in x_dim_fp32]

    plt.plot(
        num_cells_fp32, blups_fp32,
        label=f"{gpu_model} (FP32)",
        marker='o',
        linewidth=line_width,
        markersize=marker_size,
        color=colors.get(gpu_model, None),
        linestyle='-'
    )

    # optionally plot fp64 data
    if plot_fp64:
        data_fp64 = perf_data_single_fp64.get(gpu_model, [])
        if data_fp64 and not all((x == 0 and y == 0) for x, y in data_fp64):
            x_dim_fp64, blups_fp64 = zip(*data_fp64)
            num_cells_fp64 = [x**2 for x in x_dim_fp64]

            plt.plot(
                num_cells_fp64, blups_fp64,
                label=f"{gpu_model} (FP64)",
                marker='v',
                linewidth=line_width,
                markersize=marker_size,
                color=colors.get(gpu_model, None),
                linestyle=fp64_style['linestyle'],
                alpha=fp64_style['alpha']
            )

plt.grid(True, linestyle='--', alpha=1.0)
plt.legend()
plt.tight_layout()
plt.xscale('log')

# save as png
plt.savefig(outputPath, dpi=300)
plt.close()

print(f"✅  Saved plot: {outputPath}")
