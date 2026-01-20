#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_path = log_path
        f = open(self.log_path, "w")
        f.close()

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, "a") as f:
            f.write(message)

    def flush(self):
        pass
    

def load_label(label_path, test_ids):
    '''
    :param label_path: sample labels filename
    '''
    print('loading data...')
    label_df = pd.read_csv(label_path, header=0, index_col=None)
    label_df.sort_values(by='Sample_ID', ascending=True, inplace=True)
    # Train
    train_label_df = label_df[~label_df['Sample_ID'].isin(test_ids)]
    train_label_df  = train_label_df.sort_values(by='Sample_ID', ascending=True)
    # Test
    test_label_df = label_df[label_df['Sample_ID'].isin(test_ids)]
    test_label_df = test_label_df.sort_values(by='Sample_ID', ascending=True)

    return train_label_df, test_label_df


def load_data(adj_path, feature_path, label_path, threshold=0.005, keep_weights: bool = False):
    '''
    :param adj_path: the similarity matrix filename
    :param feature_path: the omics vector features filename
    :param label_path: sample labels filename
    :param threshold: the edge filter threshold
    '''
    print('loading data...')
    adj_df = pd.read_csv(adj_path, header=0, index_col=None)
    fea_df = pd.read_csv(feature_path, header=0, index_col=None)
    label_df = pd.read_csv(label_path, header=0, index_col=None)

    if adj_df.shape[0] != fea_df.shape[0] or adj_df.shape[0] != label_df.shape[0]:
        print('Input files must have same samples.')
        exit(1)

    adj_df.rename(columns={adj_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    fea_df.rename(columns={fea_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    label_df.rename(columns={label_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    
    col_samples = pd.Index([str(c) for c in adj_df.columns[1:]])
    row_samples = adj_df["Sample"].astype(str).values

    adj_df = adj_df.set_index("Sample")
    adj_df.columns = col_samples
    adj_df = adj_df.loc[row_samples, row_samples]

    fea_df = fea_df.set_index("Sample")
    label_df = label_df.set_index("Sample")

    fea_df_aligned = fea_df.loc[row_samples].reset_index()
    label_df_aligned = label_df.loc[row_samples].reset_index()

    # Threshold pruning 
    adj_m = adj_df.values.astype(float)
    if threshold is not None:
        adj_m[adj_m < threshold] = 0.0

    if not keep_weights:
        adj_m = (adj_m != 0).astype(float)

    print("Calculating the symmetrically normalized laplacian matrix...")
    n = adj_m.shape[0]
    adj_hat = adj_m + np.eye(n, dtype=float)

    deg = np.sum(adj_hat, axis=1)
    d_inv_sqrt = np.zeros_like(deg, dtype=float)
    nonzero = deg > 0
    d_inv_sqrt[nonzero] = deg[nonzero] ** (-0.5)

    D_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj_hat @ D_inv_sqrt

    print("Finished calculating.")
    return norm_adj, fea_df_aligned, label_df_aligned

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
