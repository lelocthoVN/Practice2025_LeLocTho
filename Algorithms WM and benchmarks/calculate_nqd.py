import numpy as np
from scipy.interpolate import interp1d
import json
import matplotlib.pyplot as plt


def calculate_nqd_at_value(P, Q, target_value):
    """
    Interpolates Q at P=target_value (sorted by P for stability).

    Args:
        P (np.ndarray): The x-axis data (typically TPR values).
        Q (np.ndarray): The y-axis data (typically normalized PSNR values).
        target_value (float): The specific P value at which to interpolate Q.

    Returns:
        float: The interpolated Q value, or +/- inf if the target is outside the range.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    order = np.argsort(P)
    P_sorted = P[order]
    Q_sorted = Q[order]
    if target_value >= np.max(P_sorted):
        return float("-inf")
    if target_value <= np.min(P_sorted):
        return float("inf")
    interpolator = interp1d(
        P_sorted, Q_sorted, kind='linear', fill_value="extrapolate")
    return float(interpolator(target_value))


def calculate_normalized_quality_degradation(wm_algorithms, percentile_10_, percentile_90_):
    """
    Calculates and prints the Normalized Quality Degradation (NQD) metrics.

    Args:
        wm_algorithms (list): A list of dictionaries, where each dict contains
                              data for a watermarking algorithm.
        percentile_10 (float): The 10th percentile of PSNR values for normalization.
        percentile_90 (float): The 90th percentile of PSNR values for normalization.
    """
    for wm_algorithm in wm_algorithms:
        for mask_type in wm_algorithm:  # 'no_mask', 'has_mask', ...
            for distortion_type in wm_algorithm[mask_type]:
                if str(distortion_type).startswith("Regen"):
                    continue

                psnr_arr = np.array(
                    wm_algorithm[mask_type][distortion_type]['psnr'], dtype=float)
                tpr_arr = np.array(
                    wm_algorithm[mask_type][distortion_type]['tpr'],  dtype=float)

                # Chuẩn hóa PSNR -> Q ∈ [0.1, 0.9], PSNR cao -> Q thấp
                Q = 1 - ((psnr_arr - percentile_10_) /
                         (percentile_90_ - percentile_10_) * 0.8 + 0.1)

                Q9P = calculate_nqd_at_value(tpr_arr, Q, 0.90)
                Q75P = calculate_nqd_at_value(tpr_arr, Q, 0.75)
                avg_p = float(np.mean(tpr_arr))
                avg_q = float(np.mean(Q))

                print(
                    f"Mask type: {mask_type} - Distortion: {distortion_type}")
                print(f"Q@P=0.90: {Q9P:.2f}")
                print(f"Q@P=0.75: {Q75P:.2f}")
                print(f"Avg P   : {avg_p:.2f}")
                print(f"Avg Q   : {avg_q:.2f}")
                print("-"*100)


def get_all_psnr(wm_algs):
    """
    Collects all PSNR values from the loaded data for percentile calculation.

    Args:
        wm_algs (list): A list of dictionaries, each containing a watermarking
                        algorithm's data.

    Returns:
        list: A list of all PSNR values.
    """
    psnr_list = []
    for wm_algorithm in wm_algs:
        for mask_type in wm_algorithm:
            for distortion_type in wm_algorithm[mask_type]:
                if str(distortion_type).startswith("Regen"):
                    continue
                psnr_list += wm_algorithm[mask_type][distortion_type]['psnr']
    return psnr_list


def draw_data_chart_2(json_path="data_metrics_psnr_THO.json", mask_key="no_mask"):
    """
    Draws a radar chart comparing the average TPR of different algorithms.

    Args:
        json_path (str): The path to the JSON data file.
        mask_key (str): The specific mask type to plot ('no_mask', 'has_mask', etc.).
    """
    with open(json_path, "r") as f:
        loaded_data = json.load(f)

    labels = [
        'AdjustBrightness', 'JPEG Compression', 'Rotation', 'ResizeCrop', 'Cut',
        'RandomErasing', 'AdjustContrast', 'GaussianBlur', 'GaussianNoise',
    ]
    num_vars = len(labels)

    tpr_avg = {}
    for wm in loaded_data:
        if mask_key not in loaded_data[wm]:
            continue
        tpr_avg[wm] = {}
        tpr_avg[wm][mask_key] = {}
        for dist in labels:
            if dist in loaded_data[wm][mask_key]:
                vals = loaded_data[wm][mask_key][dist]['tpr']
                tpr_avg[wm][mask_key][dist] = float(np.mean(vals))
            else:
                tpr_avg[wm][mask_key][dist] = 0.0

    if not tpr_avg:
        branches = {k: list(loaded_data[k].keys()) for k in loaded_data}
        print(f"No data found for mask_key='{mask_key}'. "
              f"Check the branches available in JSON: {branches}")
        return

    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'polar': True})
    for wm in tpr_avg:
        values = [tpr_avg[wm][mask_key][d]
                  for d in labels] + [tpr_avg[wm][mask_key][labels[0]]]
        ax.plot(angles, values, linewidth=2, label=wm)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, color='navy')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)
    plt.tight_layout()
    plt.show()


def compute_overall_psnr(loaded_data, algos=('DWT_DCT', 'DWT_DCT_SVD', 'LSB'), mask_key='no_mask'):
    """
    Computes the average PSNR for a list of algorithms under a specific mask key.

    Args:
        loaded_data (dict): The loaded JSON data.
        algos (tuple): A tuple of algorithm names to compute.
        mask_key (str): The mask type to use for computation.

    Returns:
        dict: A dictionary of {algo: avg_psnr}.
    """
    result = {}
    for a in algos:
        if a not in loaded_data or mask_key not in loaded_data[a]:
            continue
        psnr_vals = []
        for dist in loaded_data[a][mask_key]:
            psnr_vals += loaded_data[a][mask_key][dist]['psnr']
        if psnr_vals:
            result[a] = float(np.mean(psnr_vals))
    return result


def plot_overall_metric(values_dict, metric_name="Average PSNR (dB)", title=None):
    """
    Plots a bar chart to quickly compare average metric values.

    Args:
        values_dict (dict): A dictionary of {algorithm_name: metric_value}.
        metric_name (str): The name of the metric for the y-axis label.
        title (str, optional): The title of the plot.
    """
    algos = list(values_dict.keys())
    vals = [values_dict[a] for a in algos]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(algos, vals)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    ax.set_title(title or metric_name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open("data_metrics_psnr_1.json", "r") as f:
        loaded_data = json.load(f)

    # ----  Print average PSNR  ----
    for algorithm in loaded_data:  # "LSB", "DWT_DCT", "DWT_DCT_SVD"
        print(algorithm)
        for mask_status in loaded_data[algorithm]:
            psnr_curr = []
            for distortion in loaded_data[algorithm][mask_status]:
                psnr_curr += loaded_data[algorithm][mask_status][distortion]['psnr']
            print(np.average(psnr_curr))
        print("------------------")

    # ---- PLOT A BAR CHART FOR 3 AVERAGE PSNR VALUES ----
    avg_psnr = compute_overall_psnr(loaded_data,
                                    algos=('DWT_DCT', 'DWT_DCT_SVD', 'LSB'),
                                    mask_key='no_mask')
    plot_overall_metric(avg_psnr, metric_name="Average PSNR (dB)",
                        title="Overall Imperceptibility")

    # ---- calculate NQD ----
    wm_list = [loaded_data[a]
               for a in ('LSB', 'DWT_DCT', 'DWT_DCT_SVD') if a in loaded_data]

    # ---- Calculate global p10/p90 & report NQD ----
    psnr_all = get_all_psnr(wm_list)
    percentile_10 = np.percentile(psnr_all, 10)
    percentile_90 = np.percentile(psnr_all, 90)
    calculate_normalized_quality_degradation(
        wm_list, percentile_10, percentile_90)

    draw_data_chart_2("data_metrics_psnr_1.json", mask_key="no_mask")
