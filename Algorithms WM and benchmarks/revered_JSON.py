import os
import json
import pandas as pd


KNOWN_DISTORTIONS = [
    "JPEG_COMPRESSION", "CONTRAST", "CUT", "ROTATION",
    "RESIZED_CROP", "RANDOM_ERASING", "ADJUST_CONTRAST",
    "GAUSSIAN_BLUR", "GAUSSIAN_NOISE", "ADJUST_BRIGHTNESS"
]


PRETTY_MAP = {
    "JPEG_COMPRESSION": "JPEG Compression",
    "CONTRAST": "Contrast",
    "CUT": "Cut",
    "ROTATION": "Rotation",
    "RESIZED_CROP": "ResizeCrop",
    "RANDOM_ERASING": "RandomErasing",
    "ADJUST_CONTRAST": "AdjustContrast",
    "GAUSSIAN_BLUR": "GaussianBlur",
    "GAUSSIAN_NOISE": "GaussianNoise",
    "ADJUST_BRIGHTNESS": "AdjustBrightness",
}


def _parse_filename(fname: str):
    """
    From 'output_data_<ALGO>_<DISTORTION>[_HAS_MASK].csv' -> (algo, distortion, mask_key)
    Try to identify the distortion by matching the suffix with KNOWN_DISTORTIONS.
    """
    name, ext = os.path.splitext(fname)
    if not name.startswith("output_data_") or ext.lower() != ".csv":
        return None
    has_mask = name.endswith("_HAS_MASK")
    if has_mask:
        name = name[:-len("_HAS_MASK")]
    core = name[len("output_data_"):]
    hit = None
    for d in sorted(KNOWN_DISTORTIONS, key=len, reverse=True):
        if core.endswith("_" + d):
            hit = d
            break
    if hit is None:
        i = core.rfind("_")
        if i == -1:
            return None
        algo, distortion = core[:i], core[i+1:]
    else:
        distortion = hit
        algo = core[:-(len(distortion)+1)]
    mask_key = "has_mask" if has_mask else "no_mask"
    return algo, distortion, mask_key


def build_metrics_json(results_dir: str, out_json_path: str,
                       use_pretty_names: bool = True):
    """
    Scan the directory containing WAVES CSV files and merge them into a JSON:
    {
      "<ALGO>": {
        "no_mask": {
          "<DISTORTION>": {"psnr": [...], "tpr": [...]},
          ...
        },
        "has_mask": { ... }
      },
      ...
    }
    """
    data = {}

    for fname in os.listdir(results_dir):
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        algo, distortion, mask_key = parsed

        csv_path = os.path.join(results_dir, fname)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Cannot read {csv_path}: {e}")
            continue

        tpr_col = "tpr_at_0_1_fpr" if "tpr_at_0_1_fpr" in df.columns else (
                  "tpr" if "tpr" in df.columns else None)
        if "psnr_values" not in df.columns or tpr_col is None:
            print(f"[WARN] Skip {csv_path}: missing psnr/tpr columns")
            continue

        psnr_list = df["psnr_values"].astype(float).tolist()
        tpr_list = df[tpr_col].astype(float).tolist()

        if algo not in data:
            data[algo] = {}
        if mask_key not in data[algo]:
            data[algo][mask_key] = {}

        key = PRETTY_MAP.get(
            distortion, distortion) if use_pretty_names else distortion
        data[algo][mask_key][key] = {"psnr": psnr_list, "tpr": tpr_list}

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved JSON: {out_json_path}")

    print(f"  Algorithms: {list(data.keys())}")
    for algo, masks in data.items():
        for mkey, dists in masks.items():
            print(f"  - {algo} / {mkey}: {len(dists)} distortions")


if __name__ == "__main__":
    results_dir = r"E:\waves-data\main\results"
    out_json = "data_metrics_psnr_THO.json"
    build_metrics_json(results_dir, out_json, use_pretty_names=True)
