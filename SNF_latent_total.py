#!/usr/bin/env python
# -*- coding: utf-8 -*-
import snf
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg") 


def check_stats(name, views):
    print(f"\n[{name}]")
    for i, X in enumerate(views):
        print(f" modality {i} | mean={np.mean(X):.4f}, std={np.std(X):.4f}, "
              f"min={np.min(X):.4f}, max={np.max(X):.4f}")


def read_table_auto(path):
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline()
    sep = '\t' if first.count('\t') > first.count(',') else ','
    return pd.read_csv(path, sep=sep, header=0, index_col=None)

def read_latent_table_auto(path):
    total_path = os.path.join(args.mother_path, path)
    with open(total_path, 'r', encoding='utf-8') as f:
        first = f.readline()
    sep = '\t' if first.count('\t') > first.count(',') else ','
    return pd.read_csv(total_path, sep=sep, header=0, index_col=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mother_path', '-mp', type=str, required=True, help='Location of input files (mother_dir)') ##
    parser.add_argument('--train_path', '-train_p', type=str, nargs='+', required=True, help='Location of input files (Train), must be 2 or 3 files')
    parser.add_argument('--test_path', '-test_p', type=str, nargs='+', required=True, help='Location of input files (Test), must be 2 or 3 files')
    parser.add_argument('--metric', '-m', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                                           'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                                                           'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                                                           'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='sqeuclidean',
                        help='Distance metric to compute. Must be one of available metrics in :py:func scipy.spatial.distance.pdist.')
    parser.add_argument('--K', '-k', type=int, default=15,
                        help='(0, N) int, number of neighbors to consider when creating affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 20.')
    parser.add_argument('--mu', '-mu', type=float, default=0.5,
                        help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 0.5.')
    parser.add_argument('--filename', '-f', type=str, required=True, help='File name of the results')
    parser.add_argument('--subfolder', '-sf', type=str, required=True, help='Sub folder File name of the results')
    parser.add_argument('--test_list', '-tl', type=str, default='data/test_sample.csv', help='CSV with test sample IDs; columns: Sample_ID or Sample')

    args = parser.parse_args()

    if len(args.train_path) < 2 or len(args.train_path) > 3:
        raise ValueError("You must provide 2 or 3 input files for --path.")

    print('Load TRAIN/TEST merged data files...')
    omics_data_list_Train = [read_latent_table_auto(p) for p in args.train_path]
    omics_data_list_Test = [read_latent_table_auto(p) for p in args.test_path]
    omics_data_list_Merged = []

    for train_omics, test_omics in zip(omics_data_list_Train, omics_data_list_Test):
        print(train_omics.shape, test_omics.shape)
        merged_omics = pd.concat([train_omics, test_omics])
        omics_data_list_Merged.append(merged_omics)
    print("Merged Shapes:", [d.shape for d in omics_data_list_Merged])

    #save directory
    save_dir = os.path.join("result_SNF", args.filename, args.subfolder)
    os.makedirs(save_dir, exist_ok=True)

    # sample align
    for i, df in enumerate(omics_data_list_Merged):
        df.rename(columns={df.columns.tolist()[0]: 'Sample'}, inplace=True)
        df.sort_values(by='Sample', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)  

    # --- Load test list & build masks (train = not in test) ---
    test_df = read_table_auto(args.test_list)
    if 'Sample' in test_df.columns:
        test_ids = set(test_df['Sample'].astype(str).tolist())
    elif 'Sample_ID' in test_df.columns:
        test_ids = set(test_df['Sample_ID'].astype(str).tolist())
    else:
        raise ValueError("test_list must have a 'Sample' or 'Sample_ID' column.")

    # 1. Fit StandardScaler on TRAIN ONLY (exclude test) ---
    scalers, transformed_data = [], []
    train_mins, train_maxs = [], []
    for df in omics_data_list_Merged:
        tmask = df['Sample'].astype(str).isin(test_ids)
        X_train = df.loc[~tmask, df.columns[1:]].values   # (732,750)
        
        # Modality2 (CNV)의 Exernal Validation -
        train_min, train_max = X_train.min(), X_train.max()
        train_mins.append(train_min) # (수정)
        train_maxs.append(train_max) # (수정)
        
        print(f'[Train] Min: {train_min}, Max: {train_max}')
        if X_train.shape[0] == 0:
            raise ValueError("No train rows left to fit StandardScaler.")
        sc = StandardScaler().fit(X_train)
        scalers.append(sc)
        df.iloc[:, 1:] = sc.transform(df.iloc[:, 1:].values)
        transformed_data.append(df)


    # --- train/test 인덱스 생성 ---
    mask_test_all = transformed_data[0]['Sample'].astype(str).isin(test_ids)
    train_idx = np.where(~mask_test_all)[0]
    test_idx = np.where(mask_test_all)[0]

    # 2. Constant Feature mask 계산 (Train 기준)
    constant_masks = []
    for i, df in enumerate(transformed_data): ###
        X = df.iloc[:, 1:].values
        constant_mask = np.std(X[train_idx, :], axis=0) < 1e-8
        constant_masks.append(constant_mask)

        # 3. Train/Test 전체에서 해당 feature 제거
        df.iloc[:, 1:] = X[:, ~constant_mask]

    print("Removed constant features per modality:", [mask.sum() for mask in constant_masks])

    # 4. Train/Test split
    train_samples = transformed_data[0].loc[train_idx, 'Sample'].astype(str).tolist()
    test_samples = transformed_data[0].loc[test_idx, 'Sample'].astype(str).tolist()

    # 5. 뷰 분리 (모든 모달리티 동일 인덱스로 슬라이싱) ---
    train_views = []
    test_views = []
    for df in transformed_data:
        X = df.iloc[:, 1:].values.astype(np.float64)
        train_views.append(X[train_idx, :])
        test_views.append(X[test_idx, :])

    check_stats("TRAIN", train_views)
    check_stats("TEST", test_views)

    # Train/Test data 분포 check
    for i, df in enumerate(transformed_data):
        print(i, np.mean(df.iloc[:,1:].values), np.std(df.iloc[:,1:].values))

    def _keff(K, n):
        return max(1, min(K, n - 1))

        
    print('\n--- COMBINED SNF (ALL DATA) ---')
    # 1. Combine all available views and sample lists
    all_views = []
    all_samples = train_samples + test_samples 

    for i in range(len(train_views)):
        views_to_combine = [train_views[i], test_views[i]]
        all_views.append(np.concatenate(views_to_combine, axis=0))

    print("Running SNF on all combined data. Total samples:", len(all_samples))
    if len(all_samples) > 0:
        check_stats("ALL_COMBINED", all_views)
    
    # 2. Run SNF *once*
    n_all = len(all_samples)
    if n_all >= 2:
        K_all = _keff(args.K, n_all)
        print('Used K: ', K_all)
        aff_all = snf.make_affinity(all_views, metric=args.metric, K=K_all, mu=args.mu)
        fused_all = snf.snf(aff_all, K=K_all)
    else:
        fused_all = np.zeros((n_all, n_all), dtype=float)
        print("Warning: Not enough samples to run SNF.")

    # 3. Create DataFrame for splitting
    fused_all_df = pd.DataFrame(fused_all, index=all_samples, columns=all_samples)

    # 4. Split and Save TRAIN
    if train_samples:
        print("Splitting and saving TRAIN...")
        fused_tr_df = fused_all_df.loc[train_samples, train_samples]
        if fused_tr_df.shape[0] > 0:
            np.fill_diagonal(fused_tr_df.values, 0.0)
        fused_tr_df.to_csv(os.path.join(save_dir, 'SNF_fused_train.csv'), header=True, index=True)
        if fused_tr_df.shape[0] >= 2:
            gtr = sns.clustermap(fused_tr_df, cmap='magma', figsize=(8, 8))
            gtr.savefig(os.path.join(save_dir, 'SNF_fused_train.png'), dpi=300)

    # 5. Split and Save TEST
    if test_samples:
        print("Splitting and saving TEST...")
        fused_te_df = fused_all_df.loc[test_samples, test_samples]
        if fused_te_df.shape[0] > 0:
            np.fill_diagonal(fused_te_df.values, 0.0)
        fused_te_df.to_csv(os.path.join(save_dir, 'SNF_fused_test.csv'), header=True, index=True)
        if fused_te_df.shape[0] >= 2:
            gte = sns.clustermap(fused_te_df, cmap='magma', figsize=(8, 8))
            gte.savefig(os.path.join(save_dir, 'SNF_fused_test.png'), dpi=300)


    print('Success! Split SNF results saved.')