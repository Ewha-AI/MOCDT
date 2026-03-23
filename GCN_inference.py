import os
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import seaborn as sns

from model.resgcn_model import ResGCN
from utils import Logger


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def logits_to_probs(logits):
    return F.softmax(logits, dim=1)


def load_sample_list(sample_list_path):
    df = pd.read_csv(sample_list_path)

    if 'Sample_ID' in df.columns:
        sample_ids = df['Sample_ID'].astype(str).tolist()
    elif 'Sample' in df.columns:
        sample_ids = df['Sample'].astype(str).tolist()
    else:
        raise ValueError("sample_list csv must contain 'Sample_ID' or 'Sample' column.")

    return sample_ids


def load_feature_table(feature_path):
    df = pd.read_csv(feature_path)
    df.rename(columns={df.columns[0]: 'Sample'}, inplace=True)
    df['Sample'] = df['Sample'].astype(str)
    return df


def load_label_table(label_path):
    df = pd.read_csv(label_path)
    df.rename(columns={df.columns[0]: 'Sample_ID'}, inplace=True)
    df['Sample_ID'] = df['Sample_ID'].astype(str)
    return df


def load_adjacency_matrix(adj_path):
    ext = os.path.splitext(adj_path)[1].lower()

    if ext == '.npy':
        adj = np.load(adj_path)
        adj_df = None

    elif ext in ['.csv', '.txt', '.tsv']:
        sep = '\t' if ext == '.tsv' else ','
        adj_df = pd.read_csv(adj_path, index_col=0, sep=sep)
        adj_df.index = adj_df.index.astype(str)
        adj_df.columns = adj_df.columns.astype(str)
        adj = adj_df.values.astype(float)

    else:
        raise ValueError(f"Unsupported adjacency file extension: {ext}")

    return adj, adj_df


def subset_feature_and_adj(feature_df, adj_df, target_sample_ids):
    """
    feature_df: 첫 컬럼이 Sample
    adj_df: index/columns가 sample name인 adjacency DataFrame
    target_sample_ids 순서대로 subset
    """
    feature_df = feature_df.copy()
    feature_df['Sample'] = feature_df['Sample'].astype(str)

    feature_samples = set(feature_df['Sample'].tolist())
    adj_samples = set(adj_df.index.astype(str).tolist())

    missing_in_feature = [s for s in target_sample_ids if s not in feature_samples]
    missing_in_adj = [s for s in target_sample_ids if s not in adj_samples]

    if len(missing_in_feature) > 0:
        print(f"[WARN] {len(missing_in_feature)} samples from sample_list were not found in feature data.")
        print("[WARN] First few missing in feature:", missing_in_feature[:10])

    if len(missing_in_adj) > 0:
        print(f"[WARN] {len(missing_in_adj)} samples from sample_list were not found in adjacency data.")
        print("[WARN] First few missing in adjacency:", missing_in_adj[:10])

    valid_samples = [s for s in target_sample_ids if (s in feature_samples and s in adj_samples)]

    if len(valid_samples) == 0:
        raise ValueError("No overlapping valid samples found in feature data and adjacency data.")

    feature_sub = (
        feature_df.set_index('Sample')
        .loc[valid_samples]
        .reset_index()
    )

    adj_sub_df = adj_df.loc[valid_samples, valid_samples]
    adj_sub = adj_sub_df.values.astype(float)

    return feature_sub, adj_sub, valid_samples


def align_labels_to_samples(label_df, sample_ids):
    if label_df is None:
        return None

    label_map = dict(zip(label_df['Sample_ID'].astype(str), label_df.iloc[:, 1].values))
    labels = []
    missing = []

    for s in sample_ids:
        if s in label_map:
            labels.append(label_map[s])
        else:
            labels.append(np.nan)
            missing.append(s)

    if len(missing) > 0:
        print(f"[WARN] {len(missing)} samples have no label in labeldata.")
        print("[WARN] First few unlabeled samples:", missing[:10])

    return np.array(labels)


def eval_TOO(GT_origin_y_true, too_y_pred_numeric,
             save_dir=None, labels_order=None, save_name=None, verbose=True):
    acc = accuracy_score(GT_origin_y_true, too_y_pred_numeric)
    precision = precision_score(GT_origin_y_true, too_y_pred_numeric,
                                average='macro', zero_division=0)
    sensitivity = recall_score(GT_origin_y_true, too_y_pred_numeric,
                               average='macro', zero_division=0)

    if labels_order is None:
        labels_order = np.unique(np.concatenate((GT_origin_y_true, too_y_pred_numeric)))

    cm = confusion_matrix(GT_origin_y_true, too_y_pred_numeric, labels=labels_order)

    specificities = []
    for i in range(len(labels_order)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        specificities.append(specificity)
    macro_specificity = np.nanmean(specificities)

    if verbose:
        print('--- TOO Metrics (on GT-cancer among binary-predicted cancer) ---')
        print(f'Accuracy: {acc:.4f} | Precision: {precision:.4f} | '
              f'[Macro] Specificity: {macro_specificity:.4f} | Sensitivity: {sensitivity:.4f}')

    if save_dir is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
        disp.plot(cmap="Blues", colorbar=False)
        fname = save_name or "too_confmat.png"
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        if verbose:
            print(f"[TOO] Confusion matrix saved to {out_path}")
        plt.close()

    return acc, precision, sensitivity, macro_specificity, cm


def make_top_sets_with_threshold(
    probs_cancer_cond: torch.Tensor,
    cancer_class_indices: np.ndarray,
    thr_abs: float = 0.05,
    max_k: int = 8
):
    with torch.no_grad():
        k = min(max_k, probs_cancer_cond.shape[1])
        topk_probs, topk_idx_local = torch.topk(probs_cancer_cond, k=k, dim=1)
        topk_labels = np.array(cancer_class_indices)[topk_idx_local.cpu().numpy()]  # [N, k]
        topk_probs_np = topk_probs.cpu().numpy()
        N = probs_cancer_cond.shape[0]

        top_sets = [[] for _ in range(k)]

        for n in range(N):
            sample_sets = []
            p_ref = float(topk_probs_np[n, 0])
            included_labels = [int(topk_labels[n, 0])]

            sample_sets.append(included_labels.copy())

            for j in range(1, k):
                p_j = float(topk_probs_np[n, j])
                l_j = int(topk_labels[n, j])

                if abs(p_ref - p_j) < thr_abs:
                    included_labels.append(l_j)
                sample_sets.append(included_labels.copy())

            for j in range(k):
                top_sets[j].append(sample_sets[j])

        return top_sets, topk_labels, topk_probs_np


def main():
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument('--adjdata', '-ad', type=str, required=True,
                        help='Adjacency matrix path (.npy or .csv/.tsv)')
    parser.add_argument('--featuredata', '-fd', type=str, required=True,
                        help='Feature csv path')
    parser.add_argument('--sample_list', '-sl', type=str, required=True,
                        help='CSV with Sample_ID or Sample column')
    parser.add_argument('--labeldata', '-ld', type=str, default=None,
                        help='Optional label csv. If provided, metrics are computed.')

    # trained model
    parser.add_argument('--binary_model_path', type=str, required=True,
                        help='Path to best_binary_model.pkl')
    parser.add_argument('--too_model_path', type=str, required=True,
                        help='Path to best_too_model.pkl')
    parser.add_argument('--binary_threshold', type=float, required=True,
                        help='Final binary threshold found during training')

    # model config (must match training)
    parser.add_argument('--hidden', '-hd', type=int, default=64)
    parser.add_argument('--dropout', '-dp', type=float, default=0.5)
    parser.add_argument('--nclass', '-nc', type=int, default=9,
                        help='Total number of classes including normal')
    parser.add_argument('--normal_label', type=int, default=0)
    parser.add_argument('--topk_prob_gap', type=float, default=1.0,
                        help='Absolute probability gap threshold for top-k set expansion')

    # save / device
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--filename', '-f', type=str, required=True)
    parser.add_argument('--subfolder', '-sf', type=str, required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)

    save_dir = os.path.join("result_latent_CV", args.filename, args.subfolder)
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "log_inference.txt")
    stdoutOrigin = sys.stdout
    sys.stdout = Logger(log_path)

    print("[INFO] Arguments:", args)
    print("[INFO] Device:", device)


    target_sample_ids = load_sample_list(args.sample_list)
    print(f"[INFO] #Target samples from sample_list: {len(target_sample_ids)}")


    print("[INFO] Loading feature data...")
    feature_df_full = load_feature_table(args.featuredata)
    print(f"[INFO] Full feature data shape: {feature_df_full.shape}")

    print("[INFO] Loading adjacency data...")
    adj_full, adj_df_full = load_adjacency_matrix(args.adjdata)
    print(f"[INFO] Adjacency shape: {adj_full.shape}")

    if adj_full.shape[0] != adj_full.shape[1]:
        raise ValueError(f"Adjacency matrix must be square. Got {adj_full.shape}")

    if adj_full.shape[0] != feature_df_full.shape[0]:
        raise ValueError(
            f"Adjacency size and feature rows must match. "
            f"adj={adj_full.shape[0]}, features={feature_df_full.shape[0]}"
        )

    data_sub, adj_sub, valid_samples = subset_feature_and_adj(
        feature_df_full, adj_df_full, target_sample_ids
    )

    print(f"[INFO] Subset feature shape: {data_sub.shape}")
    print(f"[INFO] Subset adjacency shape: {adj_sub.shape}")
    print(f"[INFO] #Valid samples used for inference: {len(valid_samples)}")

    if len(valid_samples) == 0:
        raise ValueError("No valid samples found for inference after filtering sample_list.")

    label_arr = None
    if args.labeldata is not None:
        print("[INFO] Loading label data...")
        label_df = load_label_table(args.labeldata)
        label_arr = align_labels_to_samples(label_df, valid_samples)
        print(f"[INFO] Labels aligned. #Available labels: {np.sum(~pd.isna(label_arr))}")

    features = torch.tensor(data_sub.iloc[:, 1:].values, dtype=torch.float, device=device)
    adj = torch.tensor(adj_sub, dtype=torch.float, device=device)

    n_in = features.shape[1]
    print(f"[INFO] n_in = {n_in}")

    if not (0 <= args.normal_label < args.nclass):
        raise ValueError(f"normal_label ({args.normal_label}) out of bounds for nclass={args.nclass}.")

    cancer_class_indices = np.array([i for i in range(args.nclass) if i != args.normal_label])
    n_cancer_class = args.nclass - 1

    print("[INFO] Loading binary model...")
    GCN_bin = ResGCN(
        n_in=n_in,
        n_hid=args.hidden,
        n_out=2,
        dropout=args.dropout
    ).to(device)

    GCN_bin.load_state_dict(torch.load(args.binary_model_path, map_location=device))
    GCN_bin.eval()

    print("[INFO] Running binary inference...")
    with torch.no_grad():
        logits_bin = GCN_bin(features, adj)
        probs_bin = logits_to_probs(logits_bin)[:, 1].cpu().numpy()

    pred_bin = (probs_bin >= args.binary_threshold).astype(int)

    bin_out_df = pd.DataFrame({
        'Sample': valid_samples,
        'p_cancer': probs_bin,
        'pred_binary': pred_bin
    })

    if label_arr is not None:
        gt_binary = np.where(pd.isna(label_arr), np.nan, (label_arr != args.normal_label).astype(int))
        bin_out_df['gt_label'] = label_arr
        bin_out_df['gt_binary'] = gt_binary

    bin_out_path = os.path.join(save_dir, "inference_binary_predictions.csv")
    bin_out_df.to_csv(bin_out_path, index=False)
    print(f"[Binary] Predictions saved to {bin_out_path}")

    if label_arr is not None:
        valid_mask_metric = ~pd.isna(label_arr)
        if valid_mask_metric.sum() > 0:
            y_true_bin = (label_arr[valid_mask_metric] != args.normal_label).astype(int)
            y_pred_bin = pred_bin[valid_mask_metric]
            y_prob_bin = probs_bin[valid_mask_metric]

            cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            TN, FP, FN, TP = cm_bin.ravel()
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            acc = accuracy_score(y_true_bin, y_pred_bin)
            f1 = f1_score(y_true_bin, y_pred_bin, average='binary')

            print("\n--- Binary Classification Metrics ---")
            print(f"Accuracy= {acc:.4f}, F1-score= {f1:.4f}, "
                  f"Specificity= {spec:.4f}, Sensitivity= {sens:.4f}, "
                  f"Threshold= {args.binary_threshold:.6f}")

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Pred Normal", "Pred Cancer"],
                        yticklabels=["True Normal", "True Cancer"])
            plt.title("Binary Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(os.path.join(save_dir, "inference_binary_confmat.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

    print("[INFO] Loading TOO model...")
    GCN_too = ResGCN(
        n_in=n_in,
        n_hid=args.hidden,
        n_out=n_cancer_class,
        dropout=args.dropout
    ).to(device)

    GCN_too.load_state_dict(torch.load(args.too_model_path, map_location=device))
    GCN_too.eval()

    mask_pred_cancer = (pred_bin == 1)
    num_pred_cancer = int(mask_pred_cancer.sum())
    print(f"[INFO] #Samples predicted as cancer by binary model: {num_pred_cancer}")

    if num_pred_cancer == 0:
        print("[INFO] No samples predicted as cancer. Skip TOO inference.")
        print("\nFinished!")
        return

    features_pc = features[mask_pred_cancer]
    adj_pc = adj[mask_pred_cancer][:, mask_pred_cancer]
    samples_pc = np.array(valid_samples)[mask_pred_cancer]

    with torch.no_grad():
        logits_too = GCN_too(features_pc, adj_pc)
        probs_too = logits_to_probs(logits_too)  # [N_pred_cancer, 8]

    top_sets, topk_labels, topk_probs_np = make_top_sets_with_threshold(
        probs_cancer_cond=probs_too,
        cancer_class_indices=cancer_class_indices,
        thr_abs=args.topk_prob_gap,
        max_k=n_cancer_class
    )

    out_df = pd.DataFrame({'Sample': samples_pc})
    for i in range(topk_labels.shape[1]):
        out_df[f'top{i+1}_label'] = topk_labels[:, i]
    for i in range(topk_probs_np.shape[1]):
        out_df[f'top{i+1}_prob'] = topk_probs_np[:, i]

    if label_arr is not None:
        gt_pc = label_arr[mask_pred_cancer]
        out_df['gt_label'] = gt_pc

    out_csv = os.path.join(save_dir, "inference_TOO_topk_predictions_binaryPredCancer.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[TOO] Top-k predictions saved to {out_csv}")

    if label_arr is not None:
        gt_pc_all = label_arr[mask_pred_cancer]
        valid_labeled = ~pd.isna(gt_pc_all)
        gt_pc_all = gt_pc_all[valid_labeled].astype(int)
        probs_eval = probs_too[valid_labeled]
        samples_eval = samples_pc[valid_labeled]

        mask_gt_cancer = (gt_pc_all != args.normal_label)
        if mask_gt_cancer.sum() > 0:
            probs_eval_cancer = probs_eval[mask_gt_cancer]
            y_true_eval = gt_pc_all[mask_gt_cancer]
            samples_eval_cancer = samples_eval[mask_gt_cancer]

            with torch.no_grad():
                top1_idx_local = torch.argmax(probs_eval_cancer, dim=1)
            y_pred_eval = cancer_class_indices[top1_idx_local.cpu().numpy()]

            acc_too, prec_too, sens_too, spec_too, cm_too = eval_TOO(
                y_true_eval, y_pred_eval,
                save_dir=save_dir,
                labels_order=cancer_class_indices,
                save_name="inference_too_confmat_predCancer_GT_cancer_top1.png",
                verbose=True
            )

            prec_cls, rec_cls, _, sup_cls = precision_recall_fscore_support(
                y_true_eval, y_pred_eval,
                labels=cancer_class_indices,
                zero_division=0
            )

            print("\n[GT-cancer] Per-class precision/recall (Top-1)")
            for lab, p, r, sup in zip(cancer_class_indices, prec_cls, rec_cls, sup_cls):
                print(f"  class {lab}: precision={p:.4f} | recall={r:.4f} | support={sup}")

            df_eval = pd.DataFrame({
                "class": cancer_class_indices,
                "precision": prec_cls,
                "recall": rec_cls,
                "support": sup_cls
            })
            df_eval.to_csv(
                os.path.join(save_dir, "inference_per_class_metrics_TOO_predCancer_GT_cancer_top1.csv"),
                index=False
            )

            too_top1_df = pd.DataFrame({
                "Sample": samples_eval_cancer,
                "gt_label": y_true_eval,
                "pred_top1_label": y_pred_eval
            })
            too_top1_df.to_csv(
                os.path.join(save_dir, "inference_TOO_top1_predCancer_GT_cancer.csv"),
                index=False
            )
        else:
            print("[INFO] Among binary-predicted cancer samples, no GT-cancer samples found.")

    print("\nFinished!")


if __name__ == '__main__':
    main()
