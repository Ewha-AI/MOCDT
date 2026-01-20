import numpy as np
import pandas as pd
import argparse
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resgcn_model import ResGCN
from utils import load_data, accuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from utils import *
import sys
from collections import Counter

def compute_feature_importance(model, features, adj, node_idx, class_idx=None):
    model.eval()

    x = features.clone().detach().requires_grad_(True)
    adj_local = adj 

    logits = model(x, adj_local)  
    logits_node = logits[node_idx]  

    if class_idx is None:
        target = logits_node.argmax()
    else:
        target = torch.tensor(class_idx, device=logits_node.device, dtype=torch.long)

    logit_target = logits_node[target]

    # backward
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    logit_target.backward(retain_graph=True)

    # gradient w.r.t input features
    grads = x.grad[node_idx]       
    feats = x[node_idx].detach()    

    importance = torch.abs(grads * feats)

    importance_np = importance.detach().cpu().numpy()
    return importance_np
    
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train(model, epoch, optimizer, features, adj, labels, idx_train, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch+1:04d} | loss train: {loss_train.item():.4f} | acc train: {acc_train.item():.4f}')
    return loss_train.item()

def logits_to_probs(logits):
    return F.softmax(logits, dim=1)

def find_threshold_for_specificity(scores, y_true_binary, target_spec=0.95):
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores)
    y = np.asarray(y_true_binary)
    cand = np.unique(scores)
    best_thr, best_spec, best_sens = None, -1.0, -1.0
    for thr in cand:
        y_pred = (scores >= thr).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        TN, FP, FN, TP = cm.ravel()
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        if spec >= target_spec:
            best_thr, best_spec, best_sens = thr, spec, sens
            break
    if best_thr is None:
        best_thr = cand.max() + 1e-6
        y_pred = (scores >= best_thr).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            best_spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            best_sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return float(best_thr), float(best_spec), float(best_sens)


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
        tp = cm[i, i]; fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp; tn = cm.sum() - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        specificities.append(specificity)
    macro_specificity = np.nanmean(specificities)

    if verbose:
        print('--- TOO Metrics (on Cancer Samples) ---')
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

        for n in range(N):  # sample
            sample_sets = []
            p_ref = float(topk_probs_np[n,0])    
            included_labels = [int(topk_labels[n,0])]  # top1 label
            included_probs = [p_ref]

            sample_sets.append(included_labels.copy())

            for j in range(1, k):
                p_j = float(topk_probs_np[n,j]) 
                l_j = int(topk_labels[n,j])     # topk label

                if abs(p_ref - p_j) < thr_abs:
                    included_labels.append(l_j)
                    included_probs.append(p_j)
                sample_sets.append(included_labels.copy()) 
            
            for j in range(k):
                top_sets[j].append(sample_sets[j])

        return top_sets, topk_labels, topk_probs_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_adjdata', '-ad_tr', type=str, required=True)
    parser.add_argument('--train_featuredata', '-fd_tr', type=str, required=True)
    parser.add_argument('--train_labeldata', '-ld_tr', type=str, required=True)
    parser.add_argument('--test_adjdata', '-ad_te', type=str, required=True)
    parser.add_argument('--test_featuredata', '-fd_te', type=str, required=True)
    parser.add_argument('--test_labeldata', '-ld_te', type=str, required=True)

    parser.add_argument('--mode', '-m', type=int, choices=[2], default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--epochs', '-e', type=int, default=150)
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01)
    parser.add_argument('--hidden', '-hd', type=int, default=64)
    parser.add_argument('--dropout', '-dp', type=float, default=0.5)    
    parser.add_argument('--threshold', '-t', type=float, default=0.01)  
    parser.add_argument('--nclass', '-nc', type=int, default=9, help='Total number of classes (e.g., 8 cancers + 1 normal = 9).')
    parser.add_argument('--patience', '-p', type=int, default=20)
    parser.add_argument('--filename', '-f', type=str, required=True)
    parser.add_argument('--subfolder', '-sf', type=str, required=True)
    parser.add_argument('--normal_label', type=int, default=0, help='Original label index for the normal class.')
    parser.add_argument('--nfolds', type=int, default=4)
    parser.add_argument('--target_specificity', '-tspec', type=float, default=0.99) 
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    save_dir = os.path.join("result_latent_CV", args.filename, args.subfolder)
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join("model_pth", "GCN", args.filename, args.subfolder)
    os.makedirs(model_dir, exist_ok=True)

    # Logger
    log_path = os.path.join(save_dir, f"log.txt")
    stdoutOrigin = sys.stdout
    sys.stdout = Logger(log_path)

    print("[INFO] Arguments:", args)
    print("[INFO] Device:", device)

    print("[INFO] Loading train data...")
    adj_train_np, data_train, label_train_df = load_data(args.train_adjdata, args.train_featuredata, args.train_labeldata, args.threshold)
    print(f"[INFO] Train data loaded. Adj shape: {adj_train_np.shape}, Features shape: {data_train.shape}")
    print("[INFO] Loading test data...")
    adj_test_np, data_test, label_test_df = load_data(args.test_adjdata, args.test_featuredata, args.test_labeldata, args.threshold)
    print(f"[INFO] Test data loaded. Adj shape: {adj_test_np.shape}, Features shape: {data_test.shape}")
    feature_names = data_test.columns[1:].tolist()

    features_train = torch.tensor(data_train.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels_train_orig = torch.tensor(label_train_df.iloc[:, 1].values, dtype=torch.long, device=device)
    adj_train = torch.tensor(adj_train_np, dtype=torch.float, device=device)
    all_sample_train = data_train['Sample'].tolist()

    features_test = torch.tensor(data_test.iloc[:, 1:].values, dtype=torch.float, device=device)
    labels_test_orig = torch.tensor(label_test_df.iloc[:, 1].values, dtype=torch.long, device=device)
    adj_test = torch.tensor(adj_test_np, dtype=torch.float, device=device)
    all_sample_test = data_test['Sample'].tolist()

    assert features_train.shape[1] == features_test.shape[1], "Feature dimensions must match."

    print("[INFO] Using original 0-indexed labels for model training.")

    # Binary labels: 0 = Normal, 1 = Cancer
    if not (0 <= args.normal_label < args.nclass):
        raise ValueError(f"normal_label ({args.normal_label}) out of bounds for nclass={args.nclass}.")
    labels_train_bin = (labels_train_orig != args.normal_label).long()
    labels_test_bin = (labels_test_orig != args.normal_label).long()
    cancer_class_indices = np.array([i for i in range(args.nclass) if i != args.normal_label])

    if args.mode == 2:
        # 1) Cross-validation to find binary threshold (2-class GCN)
        print("\n[INFO] Starting cross-validation for binary threshold (2-class GCN)...")

        X_train_df = data_train.iloc[:, 1:]
        y_train_np_bin = labels_train_bin.cpu().numpy()
        skf = StratifiedKFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)

        all_val_scores = []
        all_val_labels = []

        for fold_idx, (tr_idx_local, val_idx_local) in enumerate(skf.split(X_train_df, y_train_np_bin)):
            print(f"\n[CV Fold {fold_idx + 1}/{args.nfolds}]")
            tr_idx = torch.tensor(tr_idx_local, dtype=torch.long, device=device)
            val_idx = torch.tensor(val_idx_local, dtype=torch.long, device=device)

            GCN_bin_cv = ResGCN(
                n_in=features_train.shape[1],
                n_hid=args.hidden,
                n_out=2,
                dropout=args.dropout
            ).to(device)

            optimizer = torch.optim.Adam(GCN_bin_cv.parameters(),
                                         lr=args.learningrate,
                                         weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
            )
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.3)

            for epoch in range(args.epochs):
                train(GCN_bin_cv, epoch, optimizer,
                      features_train, adj_train, labels_train_bin, tr_idx, loss_fn)
                scheduler.step()

            GCN_bin_cv.eval()
            with torch.no_grad():
                logits_all = GCN_bin_cv(features_train, adj_train)
                probs_all = logits_to_probs(logits_all)[:, 1]  
                val_scores = probs_all[val_idx].detach().cpu().numpy()
                val_labels = labels_train_bin[val_idx].cpu().numpy()

            all_val_scores.extend(val_scores.tolist())
            all_val_labels.extend(val_labels.tolist())

        all_val_scores = np.array(all_val_scores)
        all_val_labels = np.array(all_val_labels)

        final_thr, final_spec, final_sens = find_threshold_for_specificity(
            all_val_scores, all_val_labels, target_spec=args.target_specificity
        )
        print(f"\n[Final Binary Threshold] thr={final_thr:.6f} "
              f"(spec={final_spec:.3f}, sens={final_sens:.3f})")

        # 2) Train final binary model on all training data
        print("\n[INFO] Training final 2-class binary GCN on all training data...")

        GCN_bin = ResGCN(
            n_in=features_train.shape[1],
            n_hid=args.hidden,
            n_out=2,
            dropout=args.dropout
        ).to(device)

        optimizer = torch.optim.Adam(GCN_bin.parameters(),
                                     lr=args.learningrate,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.3)
        idx_train_full = torch.arange(features_train.shape[0], device=device)

        loss_values = []
        best_loss = 1e9
        best_epoch = 0
        bad_counter = 0

        for epoch in range(args.epochs):
            loss = train(GCN_bin, epoch, optimizer,
                         features_train, adj_train, labels_train_bin, idx_train_full, loss_fn)
            scheduler.step()
            loss_values.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                bad_counter = 0
                torch.save(GCN_bin.state_dict(),
                           os.path.join(model_dir, "best_binary_model.pkl"))
            else:
                bad_counter += 1

            if bad_counter >= args.patience:
                print(f"[Early Stopping - Binary] at epoch {epoch + 1}")
                break

        print(f"[INFO] Best binary model saved from epoch {best_epoch + 1}")

        # Binary training loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(loss_values, label='Binary Train Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Binary GCN Training Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "binary_training_loss.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # 3) Evaluate Binary model on Test set
        print("\n[INFO] Evaluating binary model on TEST set...")
        GCN_bin.load_state_dict(torch.load(os.path.join(model_dir, "best_binary_model.pkl"),
                                           map_location=device))
        GCN_bin.eval()
        with torch.no_grad():
            logits_test_bin = GCN_bin(features_test, adj_test)
            probs_test_bin = logits_to_probs(logits_test_bin)[:, 1].cpu().numpy()  

        y_test_true_bin = labels_test_bin.cpu().numpy()
        y_test_pred_bin = (probs_test_bin >= final_thr).astype(int)

        cm_bin = confusion_matrix(y_test_true_bin, y_test_pred_bin, labels=[0, 1])
        TN, FP, FN, TP = cm_bin.ravel()
        spec_te = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        sens_te = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        acc_te = accuracy_score(y_test_true_bin, y_test_pred_bin)
        f1_te = f1_score(y_test_true_bin, y_test_pred_bin, average='binary')

        print("\n--- Binary Classification Metrics (Cancer vs. Normal) ---")
        print(f"Accuracy= {acc_te:.4f}, F1-score= {f1_te:.4f}, "
              f"Specificity= {spec_te:.4f}, Sensitivity= {sens_te:.4f} "
              f"(with threshold= {final_thr:.6f})")

        # Save binary predictions
        bin_out_df = pd.DataFrame({
            'Sample': all_sample_test,
            'p_cancer': probs_test_bin,
            'pred_binary': y_test_pred_bin,
            'gt_binary': y_test_true_bin
        })
        bin_out_path = os.path.join(save_dir, "test_binary_predictions.csv")
        bin_out_df.to_csv(bin_out_path, index=False)
        print(f"[Binary] Predictions saved to {bin_out_path}")

        # Binary confusion matrix heatmap
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred Normal", "Pred Cancer"],
                    yticklabels=["True Normal", "True Cancer"])
        plt.title("Binary Confusion Matrix (Test)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(save_dir, "test_binary_confmat.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        
        # 4) Train separate 9-class model for TOO

        print("\n[INFO] Training separate 8-class GCN for TOO classification...")
        n_cancer_class = args.nclass - 1

        GCN_multi = ResGCN(
            n_in=features_train.shape[1],
            n_hid=args.hidden,
            n_out=n_cancer_class,
            dropout=args.dropout
        ).to(device)

        optimizer = torch.optim.Adam(GCN_multi.parameters(),
                                     lr=args.learningrate,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        loss_fn = nn.CrossEntropyLoss()

        best_loss = 1e9
        best_epoch = 0
        bad_counter = 0
        
        idx_train_cancer = (labels_train_orig != args.normal_label).nonzero(as_tuple=True)[0]
        labels_train_shifted = labels_train_orig.clone()
        labels_train_shifted[labels_train_orig != args.normal_label] -= 1

        too_loss_values = []
        for epoch in range(args.epochs):
            loss = train(GCN_multi, epoch, optimizer,
                         features_train, adj_train, labels_train_shifted, idx_train_cancer, loss_fn)
            scheduler.step()
            too_loss_values.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                bad_counter = 0
                torch.save(GCN_multi.state_dict(),
                           os.path.join(model_dir, "best_too_model.pkl"))
            else:
                bad_counter += 1

            if bad_counter >= args.patience:
                print(f"[Early Stopping - TOO] at epoch {epoch + 1}")
                break

        print(f"[INFO] Best TOO model saved from epoch {best_epoch + 1}")

        # TOO training loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(too_loss_values, label='TOO Train Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("TOO GCN Training Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "too_training_loss.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # 5) TOO Evaluation ONLY for binary-predicted cancer
        print("\n[INFO] Evaluating TOO model (only on binary-predicted cancer samples)...")
        GCN_multi.load_state_dict(torch.load(os.path.join(model_dir, "best_too_model.pkl"),
                                             map_location=device))
        GCN_multi.eval()
        with torch.no_grad():
            logits_test_multi = GCN_multi(features_test, adj_test)
            probs_test_all = logits_to_probs(logits_test_multi)  # [N, 8]

        # Mask: samples predicted as cancer by binary model
        mask_pred_cancer = (y_test_pred_bin == 1)
        num_pred_cancer = int(mask_pred_cancer.sum())
        print(f"[INFO] #Samples predicted as cancer (binary): {num_pred_cancer}")

        if num_pred_cancer > 0:
            probs_pc_cond = probs_test_all[mask_pred_cancer]
            
            y_true_pc_all = labels_test_orig[mask_pred_cancer].cpu().numpy()
            samples_pc_all = np.array(all_sample_test)[mask_pred_cancer]
            
            mask_gt_cancer = (y_true_pc_all != args.normal_label)
            if mask_gt_cancer.sum() > 0:
                probs_eval = probs_pc_cond[mask_gt_cancer]              
                y_true_eval = y_true_pc_all[mask_gt_cancer]             # GT (Original 1~8)
                samples_eval = samples_pc_all[mask_gt_cancer]

                with torch.no_grad():
                    top1_idx_local = torch.argmax(probs_eval, dim=1)
                y_pred_eval = cancer_class_indices[top1_idx_local.cpu().numpy()]  

                acc_too, prec_too, sens_too, spec_too, cm_too = eval_TOO(
                    y_true_eval, y_pred_eval,
                    save_dir=save_dir,
                    labels_order=cancer_class_indices,
                    save_name="too_confmat_predCancer_GT_cancer_top1.png",
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
                df_eval.to_csv(os.path.join(save_dir,
                                            "per_class_metrics_TOO_predCancer_GT_cancer_top1.csv"),
                               index=False)
            else:
                print("[INFO] Among binary-predicted cancer, no GT-cancer samples; skip TOO metric.")

            top_sets, topk_labels, topk_probs_np = make_top_sets_with_threshold(
                probs_cancer_cond=probs_pc_cond,
                cancer_class_indices=cancer_class_indices,
                thr_abs=1,  ##
                max_k=8
            )

            y_true_pc_full = y_true_pc_all
            def compute_topk_accuracy(top_sets_list, true_labels):
                accs = []
                for sets_k in top_sets_list:
                    correct = sum(int(t) in s for t, s in zip(true_labels, sets_k))
                    accs.append(correct / len(true_labels))
                return accs

            topk_accs = compute_topk_accuracy(top_sets, y_true_pc_full)
            print("\n[Top-k accuracy on binary-predicted cancer samples (including GT-normal)]")
            for k_idx, acc_k in enumerate(topk_accs[:3], start=1):
                print(f"Top-{k_idx}: {acc_k:.4f}")

            # --- per-class precision/recall --- 
            prec_cls_pc, rec_cls_pc, _, support_pc = precision_recall_fscore_support(
                y_true_pc_full, topk_labels[:,0],
                labels=cancer_class_indices, zero_division=0
            )
            print("\n[Predicted-cancer ALL] Per-class precision/recall (Top-1)")
            for lab, p, r, sup in zip(cancer_class_indices, prec_cls_pc, rec_cls_pc, support_pc):
                print(f"  class {lab}: precision={p:.4f} | recall={r:.4f} | support={sup}")

            df_pc = pd.DataFrame({
                "class": cancer_class_indices,
                "precision": prec_cls_pc,
                "recall": rec_cls_pc,
                "support": support_pc
            })
            df_pc.to_csv(os.path.join(save_dir, "per_class_metrics_PredCancerALL_top1.csv"), index=False)

            max_k_eff = topk_labels.shape[1]
            out_df = pd.DataFrame({'Sample': samples_pc_all,
                                   'gt_label': y_true_pc_full})
            for i in range(max_k_eff):
                out_df[f'top{i+1}_label'] = topk_labels[:, i]
            for i in range(max_k_eff):
                out_df[f'top{i+1}_prob'] = topk_probs_np[:, i]

            out_csv = os.path.join(save_dir,
                                   "Test_TOO_topk_predictions_binaryPredCancer.csv")
            out_df.to_csv(out_csv, index=False)
            print(f"[TOO] Top-k info saved to {out_csv}")
            
        else:
            print("[INFO] No samples predicted as cancer by binary model; skip TOO stage on test set.")

    print("\nFinished!")
