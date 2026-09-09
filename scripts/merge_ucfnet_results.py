#!/usr/bin/env python3
"""Merge UCF-Net eval shards into portable result files."""

import argparse
import json
import os

import numpy as np
from sklearn import metrics



IN_KEYS = ("CDF", "DFFD", "DFDCP", "FF++ (c40)", "DF40", "MFFI", "mAUC")
BASE_KEYS = ("FS", "FR", "FE", "EFS", "AVG")
CROSS_KEYS = ("UADFV", "DFF", "DFDC", "FS", "FR", "EFS", "FE", "mAUC")


def load_shards(paths):
    chunks = []
    for path in sorted(paths):
        data = np.load(path, allow_pickle=True)
        chunks.append({
            "pred": data["pred"],
            "label": data["label"],
            "dataset": data["dataset"],
            "fine_label": data["fine_label"],
            "valid_image": data["valid_image"] if "valid_image" in data.files else np.ones_like(data["label"], dtype=np.bool_),
            "batch_size": int(data["batch_size"][0]),
            "trainable_params": int(data["trainable_params"][0]),
            "total_params": int(data["total_params"][0]),
        })
    merged = {}
    for key in ("pred", "label", "dataset", "fine_label", "valid_image"):
        merged[key] = np.concatenate([chunk[key] for chunk in chunks])
    merged["batch_size"] = chunks[0]["batch_size"]
    merged["trainable_params"] = chunks[0]["trainable_params"]
    merged["total_params"] = chunks[0]["total_params"]
    return merged


def fmt_percent(value):
    return "{:.2f}".format(100.0 * float(value))


def mean(values):
    return float(sum(values) / len(values))


def auc_for(data, mask):
    labels = data["label"][mask].astype(np.int8)
    preds = data["pred"][mask].astype(np.float64)
    if len(labels) == 0:
        raise ValueError("Empty subset for AUC")
    if len(np.unique(labels)) != 2:
        raise ValueError("AUC subset must contain both classes")
    return float(metrics.roc_auc_score(labels, preds))


def acc_for(data, mask):
    labels = data["label"][mask].astype(np.int8)
    preds = data["pred"][mask].astype(np.float64)
    if len(labels) == 0:
        raise ValueError("Empty subset for ACC")
    return float(((preds > 0.5).astype(np.int8) == labels).mean())


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def safe_mean(values):
    valid = [value for value in values if not np.isnan(value)]
    if not valid:
        return float("nan")
    return mean(valid)


def json_sanitize(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitize(item) for item in value]
    return value


def safe_auc_for(data, mask):
    try:
        return auc_for(data, mask)
    except ValueError:
        return float("nan")


def safe_acc_for(data, mask):
    try:
        return acc_for(data, mask)
    except ValueError:
        return float("nan")


def compute_cross_domain_partial(data):
    dataset = data["dataset"].astype(str)
    fine_label = data["fine_label"].astype(str)
    label = data["label"].astype(np.int8)

    values = {
        "UADFV": safe_auc_for(data, dataset == "UADFV"),
        "DFF": safe_auc_for(data, dataset == "DFF"),
        "DFDC": safe_auc_for(data, dataset == "DFDC"),
    }
    df40_test_real = (dataset == "DF40-test") & (label == 0)
    for source in ("FS", "FR", "EFS", "FE"):
        mask = df40_test_real | ((dataset == "DF40-test") & (label == 1) & (fine_label == source))
        values[source] = safe_auc_for(data, mask)
    values["mAUC"] = safe_mean([values[key] for key in CROSS_KEYS[:-1]])

    return {
        "cross_domain": values,
        "samples": int(len(label)),
        "batch_size": data["batch_size"],
        "trainable_params": data["trainable_params"],
        "total_params": data["total_params"],
        "invalid_images": int((~data["valid_image"].astype(np.bool_)).sum()),
    }


def compute_cross_domain(data):
    dataset = data["dataset"].astype(str)
    fine_label = data["fine_label"].astype(str)
    label = data["label"].astype(np.int8)

    values = {
        "UADFV": auc_for(data, dataset == "UADFV"),
        "DFF": auc_for(data, dataset == "DFF"),
        "DFDC": auc_for(data, dataset == "DFDC"),
    }
    df40_test_real = (dataset == "DF40-test") & (label == 0)
    for source in ("FS", "FR", "EFS", "FE"):
        mask = df40_test_real | ((dataset == "DF40-test") & (label == 1) & (fine_label == source))
        values[source] = auc_for(data, mask)
    values["mAUC"] = mean([values[key] for key in CROSS_KEYS[:-1]])

    return {
        "cross_domain": values,
        "samples": int(len(label)),
        "batch_size": data["batch_size"],
        "trainable_params": data["trainable_params"],
        "total_params": data["total_params"],
        "invalid_images": int((~data["valid_image"].astype(np.bool_)).sum()),
    }


def compute_tables_partial(data):
    dataset = data["dataset"].astype(str)
    fine_label = data["fine_label"].astype(str)
    label = data["label"].astype(np.int8)
    real_all = label == 0

    base_values = {}
    for source in ("FS", "FR", "FE", "EFS"):
        mask = real_all | ((label == 1) & (fine_label == source))
        base_values[source] = safe_acc_for(data, mask)
    base_values["AVG"] = safe_mean([base_values[x] for x in ("FS", "FR", "FE", "EFS")])

    cross = compute_cross_domain_partial(data)["cross_domain"]
    in_values = {
        "CDF": safe_auc_for(data, (dataset == "Celeb-DF-v1") | (dataset == "Celeb-DF-v2")),
        "DFFD": safe_auc_for(data, dataset == "DFFD"),
        "DFDCP": safe_auc_for(data, dataset == "DFDCP"),
        "FF++ (c40)": safe_auc_for(data, dataset == "FF++(c40)"),
        "DF40": safe_auc_for(data, dataset == "DF40"),
        "MFFI": safe_auc_for(data, dataset == "MFFI"),
    }
    in_values["mAUC"] = safe_mean([in_values[x] for x in ("CDF", "DFFD", "DFDCP", "FF++ (c40)", "DF40", "MFFI")])

    return {
        "base_model": base_values,
        "cross_domain": cross,
        "in_domain": in_values,
        "samples": int(len(label)),
        "batch_size": data["batch_size"],
        "trainable_params": data["trainable_params"],
        "total_params": data["total_params"],
        "invalid_images": int((~data["valid_image"].astype(np.bool_)).sum()),
    }


def compute_tables(data):
    dataset = data["dataset"].astype(str)
    fine_label = data["fine_label"].astype(str)
    label = data["label"].astype(np.int8)
    real_all = label == 0

    base_values = {}
    for source in ("FS", "FR", "FE", "EFS"):
        mask = real_all | ((label == 1) & (fine_label == source))
        base_values[source] = acc_for(data, mask)
    base_values["AVG"] = mean([base_values[x] for x in ("FS", "FR", "FE", "EFS")])

    cross = compute_cross_domain(data)["cross_domain"]
    in_values = {
        "CDF": auc_for(data, (dataset == "Celeb-DF-v1") | (dataset == "Celeb-DF-v2")),
        "DFFD": auc_for(data, dataset == "DFFD"),
        "DFDCP": auc_for(data, dataset == "DFDCP"),
        "FF++ (c40)": auc_for(data, dataset == "FF++(c40)"),
        "DF40": auc_for(data, dataset == "DF40"),
        "MFFI": auc_for(data, dataset == "MFFI"),
    }
    in_values["mAUC"] = mean([in_values[x] for x in ("CDF", "DFFD", "DFDCP", "FF++ (c40)", "DF40", "MFFI")])

    return {
        "base_model": base_values,
        "cross_domain": cross,
        "in_domain": in_values,
        "samples": int(len(label)),
        "batch_size": data["batch_size"],
        "trainable_params": data["trainable_params"],
        "total_params": data["total_params"],
        "invalid_images": int((~data["valid_image"].astype(np.bool_)).sum()),
    }


def compute_in_domain_only(data):
    dataset = data["dataset"].astype(str)
    label = data["label"].astype(np.int8)

    values = {
        "CDF": auc_for(data, (dataset == "Celeb-DF-v1") | (dataset == "Celeb-DF-v2")),
        "DFFD": auc_for(data, dataset == "DFFD"),
        "DFDCP": auc_for(data, dataset == "DFDCP"),
        "FF++ (c40)": auc_for(data, dataset == "FF++(c40)"),
        "DF40": auc_for(data, dataset == "DF40"),
        "MFFI": auc_for(data, dataset == "MFFI"),
    }
    values["mAUC"] = mean([values[x] for x in ("CDF", "DFFD", "DFDCP", "FF++ (c40)", "DF40", "MFFI")])

    return {
        "in_domain": values,
        "samples": int(len(label)),
        "batch_size": data["batch_size"],
        "trainable_params": data["trainable_params"],
        "total_params": data["total_params"],
        "invalid_images": int((~data["valid_image"].astype(np.bool_)).sum()),
    }


def table_rows(item, model_name, detector_type, eval_mode):
    rows = {}
    if "cross_domain" in item:
        rows["cross_domain"] = "{} & {} & {} \\\\".format(
            model_name,
            detector_type,
            " & ".join(fmt_percent(item["cross_domain"][key]) for key in CROSS_KEYS),
        )
    if "in_domain" in item:
        rows["in_domain"] = "{} & {} & {} \\\\".format(
            model_name,
            detector_type,
            " & ".join(fmt_percent(item["in_domain"][key]) for key in IN_KEYS),
        )
    if "base_model" in item:
        rows["base_model"] = "{} & {} \\\\".format(
            model_name,
            " & ".join(fmt_percent(item["base_model"][key]) for key in BASE_KEYS),
        )
    return rows


def write_portable_summary(path, item, model_name, eval_mode):
    lines = ["# {} {} evaluation".format(model_name, eval_mode), ""]
    lines.extend(
        [
            "| Samples | Invalid Images | Batch | Trainable Params | Total Params |",
            "|---:|---:|---:|---:|---:|",
            "| {} | {} | {} | {} | {} |".format(
                item["samples"],
                item["invalid_images"],
                item["batch_size"],
                item["trainable_params"],
                item["total_params"],
            ),
            "",
        ]
    )
    if "cross_domain" in item:
        cross = item["cross_domain"]
        lines.extend(
            [
                "## Cross-domain AUC",
                "| UADFV | DFF | DFDC | FS | FR | EFS | FE | mAUC |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| {} |".format(" | ".join(fmt_percent(cross[key]) for key in CROSS_KEYS)),
                "",
            ]
        )
    if "in_domain" in item:
        ind = item["in_domain"]
        lines.extend(
            [
                "## In-domain AUC",
                "| CDF | DFFD | DFDCP | FF++ (c40) | DF40 | MFFI | mAUC |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                "| {} |".format(" | ".join(fmt_percent(ind[key]) for key in IN_KEYS)),
                "",
            ]
        )
    if "base_model" in item:
        base = item["base_model"]
        lines.extend(
            [
                "## Base-model ACC",
                "| FS | FR | FE | EFS | AVG |",
                "|---:|---:|---:|---:|---:|",
                "| {} |".format(" | ".join(fmt_percent(base[key]) for key in BASE_KEYS)),
                "",
            ]
        )
    write(path, "\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-raw", nargs="+", required=True)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-name", default="UCF-Net")
    parser.add_argument("--detector-type", default="CLIP-DINO")
    parser.add_argument("--eval-mode", choices=("cross-domain", "in-domain", "all"), default="cross-domain")
    parser.add_argument("--skip-sample-check", action="store_true")
    parser.add_argument("--allow-partial-metrics", action="store_true")
    args = parser.parse_args()

    data = load_shards(args.model_raw)
    sample_count = len(data["label"])
    if not args.skip_sample_check and args.expected_samples is not None and sample_count != args.expected_samples:
        raise ValueError("{} samples {} != {}".format(args.model_name, sample_count, args.expected_samples))

    if args.eval_mode == "cross-domain":
        item = compute_cross_domain_partial(data) if args.allow_partial_metrics else compute_cross_domain(data)
        metrics_name = "cross_domain_metrics.json"
    elif args.eval_mode == "all":
        item = compute_tables_partial(data) if args.allow_partial_metrics else compute_tables(data)
        metrics_name = "all_metrics.json"
    else:
        if args.allow_partial_metrics:
            partial = compute_tables_partial(data)
            item = {
                "in_domain": partial["in_domain"],
                "samples": partial["samples"],
                "batch_size": partial["batch_size"],
                "trainable_params": partial["trainable_params"],
                "total_params": partial["total_params"],
                "invalid_images": partial["invalid_images"],
            }
        else:
            item = compute_in_domain_only(data)
        metrics_name = "in_domain_metrics.json"

    rows = table_rows(item, args.model_name, args.detector_type, args.eval_mode)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    metrics_json = json.dumps(json_sanitize(item), indent=2, sort_keys=True) + "\n"
    row_json = json.dumps(
        {
            "model_name": args.model_name,
            "detector_type": args.detector_type,
            "eval_mode": args.eval_mode,
            "samples": sample_count,
            "rows": rows,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"

    for directory in (args.output_dir, args.results_dir):
        write(os.path.join(directory, metrics_name), metrics_json)
        write(os.path.join(directory, "table_row.json"), row_json)
        write_portable_summary(os.path.join(directory, "summary.md"), item, args.model_name, args.eval_mode)
        row_lines = ["# {} table row".format(args.model_name), ""]
        for name, row in rows.items():
            row_lines.extend(["## {}".format(name), "", "```tex", row, "```", ""])
        write(os.path.join(directory, "table_row.md"), "\n".join(row_lines).rstrip() + "\n")

    print("Wrote {} results to {} and {}".format(args.model_name, args.output_dir, args.results_dir))


if __name__ == "__main__":
    main()
