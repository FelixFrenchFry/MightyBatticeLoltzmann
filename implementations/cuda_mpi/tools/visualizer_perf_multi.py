#!/home/felix/code/MightyBatticeLoltzmann/.venv/bin/python
import matplotlib
matplotlib.use('Agg') # fast backend
import matplotlib.pyplot as plt



# benchmarks for X=30000 * Y=30000
# SOURCE: cuda_mpi_04/benchmarks_fp32.txt
perf_data_multi_30k = {
    'H100': [(1, 28.6), (2, 56.7), (3, 84.5), (4, 112.7), (6, 168.3),
             (8, 222.5), (12, 335.1), (16, 431.2), (20, 540.3), (24, 645.1)],
    'A100': [(1, 20.5), (2, 41.2),             (4, 80.8), (6, 122.1),
             (8, 162.7), (12, 243.0), (16, 322.9),              (24, 474.0)]
}

# benchmarks for X=60000 * Y=60000
# SOURCE: cuda_mpi_04/benchmarks_fp32.txt
perf_data_multi_60k = {
    'H100': [(8, 227.1), (12, 337.6), (16, 451.9), (20, 565.1), (24, 674.7), (32, 888.6)]
}

# benchmarks for X=30000 * Y=30000
# TODO
perf_data_multi_async_30k = {

}

# style
font_size = 18
line_width = 2.5
marker_size = 10
colors = {
    'H100': '#ff0032',
    'H100_60k': '#780b0b',
    'A100': '#2846b4'
}
dataset_styles = {
    '30k': {
        'linestyle': '-',
        'marker': 'o',
        'alpha': 1.0,
    },
    '60k': {
        'linestyle': '-',
        'marker': 's',
        'alpha': 1.0,
    }
}
ideal_line_style = 'dotted'
ideal_alpha = 1.0

# mode selection
mode = "30k"
plot_60k = False

outputPath = "output/bench/bench_multi_30k" + ("_60k" if plot_60k else "") + ".png"

# plotting
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': font_size})


# x-axis ticks
all_gpu_counts = sorted({g for data in perf_data_multi_30k.values() for g, _ in data})
if plot_60k:
    all_gpu_counts += [g for data in perf_data_multi_60k.values() for g, _ in data]
    all_gpu_counts = sorted(set(all_gpu_counts))
plt.xticks(all_gpu_counts + [32])

def plot_dataset(perf_data, label_suffix, style_key):
    for gpu_model, data in perf_data.items():
        gpus, blups = zip(*data)

        # actual performance line
        plt.plot(
            gpus, blups,
            label = f"{gpu_model} (60,000 * 60,000)" if label_suffix == "60k" else gpu_model,
            marker=dataset_styles[style_key]['marker'],
            linestyle=dataset_styles[style_key]['linestyle'],
            linewidth=line_width,
            markersize=marker_size,
            alpha=dataset_styles[style_key]['alpha'],
            color=colors.get(f"{gpu_model}_{label_suffix}", colors.get(gpu_model, None))
        )

        # ideal scaling line
        if not (gpu_model == "H100" and label_suffix == "60k"):
            baseline_gpu = gpus[0]
            baseline_blups = blups[0]
            max_gpu = max(all_gpu_counts)
            ideal_gpus = sorted(set(g for g in all_gpu_counts if g >= baseline_gpu))
            ideal_blups = [baseline_blups * (g / baseline_gpu) for g in ideal_gpus]
            plt.plot(
                ideal_gpus, ideal_blups,
                linestyle=ideal_line_style,
                linewidth=line_width,
                alpha=ideal_alpha,
                label=f"{gpu_model} (ideal)",
                color=colors.get(gpu_model, None)
            )

# plot 30k base
plot_dataset(perf_data_multi_30k, "30k", "30k")

# (optional) overlay 60k
if plot_60k:
    plot_dataset(perf_data_multi_60k, "60k", "60k")

plt.grid(True, linestyle='--', alpha=1.0)

handles, labels = plt.gca().get_legend_handles_labels()

def sort_key(label):
    if "H100 (60,000 * 60,000)" in label:
        return (0, label)
    elif label == "H100":
        return (1, label)
    elif label == "A100":
        return (2, label)
    elif label == "H100 (ideal)":
        return (3, label)
    elif label == "A100 (ideal)":
        return (4, label)
    else:
        return (5, label)

sorted_pairs = sorted(zip(labels, handles), key=lambda pair: sort_key(pair[0]))
sorted_labels, sorted_handles = zip(*sorted_pairs)
plt.legend(sorted_handles, sorted_labels)

plt.tight_layout()

# save as png
plt.savefig(outputPath, dpi=300)
plt.close()

print(f"✅  Saved plot: {outputPath}")
