import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import model.Contrastive_Triplet_Supervised_Adversarial_AE_model_cv
import torch
import torch.utils.data as Data
import os 
import math
from sklearn.preprocessing import StandardScaler
from utils import *
from collections import Counter
import sys
import joblib
from sklearn.model_selection import train_test_split

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def save_latent_to_dataframe(latent_data, sample_name, save_path):
    latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
    latent_df.insert(0, 'Sample', sample_name)
    latent_df.to_csv(save_path, index=False)

    print(f"[INFO] Saved latent to {save_path}")
    print(latent_df.shape)


def save_latent(mmae, TX, sample_name, in_feas, save_path):
    mmae.eval()
    f0, f1, f2 = in_feas
    omics_1 = TX[:, :f0]
    omics_2 = TX[:, f0:f0+f1]
    omics_3 = TX[:, f0+f1:f0+f1+f2]
    latent_data, _, _, _, logits, latent_z1, latent_z2, latent_z3 = mmae.forward(omics_1, omics_2, omics_3)
    '''save fused latent'''
    save_latent_to_dataframe(latent_data, sample_name, save_path+'.csv')
    '''save per modality'''
    save_latent_to_dataframe(latent_z1, sample_name, save_path+'_met.csv')
    save_latent_to_dataframe(latent_z2, sample_name, save_path+'_fsr.csv')
    save_latent_to_dataframe(latent_z3, sample_name, save_path+'_cnv.csv')


def work(data, label_train_df, label_val_df, label_test_df, in_feas, lr=0.001, bs=32, epochs=100, device=torch.device('cpu'), a=0.4, b=0.3, c=0.3, mode=0, topn=100):
    '''data: Merged_data (Train+Test)'''
    train_data = pd.merge(label_train_df[['Sample_ID']],  data, left_on='Sample_ID', right_on='Sample',how='left')
    val_data = pd.merge(label_val_df[['Sample_ID']],  data, left_on='Sample_ID', right_on='Sample',how='left') 
    test_data  = data[data['Sample'].isin(label_test_df['Sample_ID'])]
    train_data = train_data[['Sample'] + [col for col in data.columns if col != 'Sample']]
    val_data = val_data[['Sample'] + [col for col in data.columns if col != 'Sample']]

    print('\nData-Label Check!')
    print('** Train split **')
    print(train_data)
    print(label_train_df)
    print((train_data['Sample'].values == label_train_df['Sample_ID'].values).sum())
    print('** Val split **')
    print(val_data)
    print(label_val_df)
    print((val_data['Sample'].values == label_val_df['Sample_ID'].values).sum())
    print('** Test **')
    print(test_data)
    print(label_test_df)
    print(len(set(test_data['Sample'])&set(label_test_df['Sample_ID'])))

    # Train: train/val split
    X_train = train_data.iloc[:, 1:].to_numpy()
    Y_train = label_train_df.iloc[:, 1].to_numpy()
    X_val = val_data.iloc[:, 1:].to_numpy()
    Y_val = label_val_df.iloc[:, 1].to_numpy()
    # Test
    X_test = test_data.iloc[:, 1:].to_numpy()
    trainval_data = pd.concat([train_data, val_data])
    print(trainval_data)

    ## if Outlier exists ##
    # X_external_val = np.clip(X_external_val, trainval_data.iloc[:,1:].values.min(), trainval_data.iloc[:,1:].values.max()) 
    # print(f'[Outlier] Min: {X_external_val.min()}  Max: {X_external_val.max()}')
    print('Data shape:', X_train.shape, X_val.shape, X_test.shape)

    print(Counter(Y_val))

    # Tensor
    TX_train, TY_train = torch.tensor(X_train, dtype=torch.float, device=device), torch.tensor(Y_train, dtype=torch.long, device=device)
    TX_val, TY_val = torch.tensor(X_val, dtype=torch.float, device=device), torch.tensor(Y_val, dtype=torch.long, device=device)
    TX_test = torch.tensor(X_test, dtype=torch.float, device=device)

    model_dir = os.path.join("model_pth/AE", args.filename, args.subfolder)
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pt")

    ## Train
    if mode == 0:
        print('\nTraining model...')
        train_Tensor_data = Data.TensorDataset(TX_train, TY_train)
        val_Tensor_data = Data.TensorDataset(TX_val, TY_val)
        train_loader = Data.DataLoader(train_Tensor_data, batch_size=bs, shuffle=True)
        val_loader = Data.DataLoader(val_Tensor_data, batch_size=bs, shuffle=True)

        mmae = model.Contrastive_Triplet_Supervised_Adversarial_AE_model_cv.Contrastive_SV_Adversarial_MMAE(in_feas, latent_dim=args.latent, num_classes=9, a=a, b=b, c=c, lambda_adv=args.lambda_adv, lambda_con=args.lambda_con) ##
        mmae.to(device)
        mmae.train()
        mmae.train_MMAE(model_dir, save_dir, train_loader, val_loader, learning_rate=lr, device=device, epochs=epochs)
        mmae.eval()       


    ## Load saved model & Inference
    if mode in [0,2]:
        print('\nGet the latent layer output...')
        mmae = model.Contrastive_Triplet_Supervised_Adversarial_AE_model_cv.Contrastive_SV_Adversarial_MMAE(in_feas, latent_dim=args.latent, num_classes=9, a=a, b=b, c=c, lambda_adv=args.lambda_adv, lambda_con=args.lambda_con) ##
        mmae.load_state_dict(torch.load(best_model_path, map_location=device)) 
        mmae.to(device)
        mmae.eval()

        if mode==0:   # Train
            out_name = "latent_train"
            TX_Train = torch.cat([TX_train, TX_val], dim=0)
            save_latent(mmae, TX_Train, train_data['Sample'].tolist() + val_data['Sample'].tolist(), in_feas, os.path.join(save_dir, out_name))

        else:          # Val, Test
            print(f"\n[INFO] Generating latent for Testset...")
            out_name = f"latent_test"
            save_latent(mmae, TX_test, test_data['Sample'].tolist(), in_feas, os.path.join(save_dir, out_name))

    return 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=int, choices=[0,1,2], default=0,
                        help='Mode 0: train&intagrate, Mode 1: just train, Mode 2: just intagrate, default: 0.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--path1', '-p1', type=str, required=True, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', type=str, required=True, help='The second omics file name.') 
    parser.add_argument('--path3', '-p3', type=str, required=True, help='The third omics file name.')
    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='Training batchszie, default: 32.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Training epochs, default: 100.')
    parser.add_argument('--latent', '-l', type=int, default=100, help='The latent layer dim, default: 100.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu', help='Training on cpu or gpu, default: cpu.')
    parser.add_argument('--a', '-a', type=float, default=0.2, help='[0,1], float, weight for the first omics data')
    parser.add_argument('--b', '-b', type=float, default=0.4, help='[0,1], float, weight for the second omics data.')
    parser.add_argument('--c', '-c', type=float, default=0.4, help='[0,1], float, weight for the third omics data.')
    parser.add_argument('--topn', '-n', type=int, default=100, help='Extract top N features every 10 epochs, default: 100.')
    parser.add_argument('--filename', '-f', type=str, required=True, help='File name of the results')
    parser.add_argument('--subfolder', '-sf', type=str, required=True, help='Sub folder File name of the results')
    parser.add_argument('--test_list', '-tl', type=str, default='data/test_sample.csv', help='CSV with test sample IDs; columns: Sample_ID or Sample')
    parser.add_argument('--tag', '-t', type=str, default=None, help='Suffix for latent csv filename, e.g., train or test')
    parser.add_argument('--labeldata', '-ld', type=str, required=True)
    parser.add_argument('--lambda_cls', '-lambda_cls', type=float, required=True)
    parser.add_argument('--lambda_adv', '-lambda_adv', type=float, required=True)
    parser.add_argument('--lambda_con', '-lambda_con', type=float, required=True)
    args = parser.parse_args()

    #read data
    omics_data1 = pd.read_csv(args.path1, header=0, index_col=None)
    omics_data2 = pd.read_csv(args.path2, header=0, index_col=None)
    omics_data3 = pd.read_csv(args.path3, header=0, index_col=None)

    save_dir = os.path.join("result_latent_CV", args.filename, args.subfolder)
    os.makedirs(save_dir, exist_ok=True)

    scaler1_path = os.path.join(save_dir, "scaler_omics1.pkl")
    scaler2_path = os.path.join(save_dir, "scaler_omics2.pkl")
    scaler3_path = os.path.join(save_dir, "scaler_omics3.pkl")
    
    # Set Logger    
    log_mode_dict = {0:'Train', 2:'Test'}
    log_path = os.path.join(save_dir, f"log_{log_mode_dict[args.mode]}.txt")
    stdoutOrigin = sys.stdout
    sys.stdout = Logger(log_path)

    #Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #set random seed
    setup_seed(args.seed)

    if args.a + args.b + args.c != 1.0:
        print('The sum of weights must be 1.')
        exit(1)

    #dims of each omics data
    in_feas = [omics_data1.shape[1] - 1, omics_data2.shape[1] - 1, omics_data3.shape[1] - 1]
    omics_data1.rename(columns={omics_data1.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data2.rename(columns={omics_data2.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data3.rename(columns={omics_data3.columns.tolist()[0]: 'Sample'}, inplace=True)

    omics_data1.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data2.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data3.sort_values(by='Sample', ascending=True, inplace=True)

    test_df = pd.read_csv(args.test_list)
    if 'Sample' in test_df.columns:
        test_ids = set(test_df['Sample'].astype(str).tolist())
    elif 'Sample_ID' in test_df.columns:
        test_ids = set(test_df['Sample_ID'].astype(str).tolist())
    else:
        raise ValueError("test_list must have a 'Sample' or 'Sample_ID' column.")

    tmask1 = omics_data1['Sample'].astype(str).isin(test_ids)
    tmask2 = omics_data2['Sample'].astype(str).isin(test_ids)
    tmask3 = omics_data3['Sample'].astype(str).isin(test_ids)

    print('Test ID:', len(test_ids), list(test_ids)[:5],'...')
    print(f"[TEST IDs] in omics1/2/3: {tmask1.sum()}/{tmask2.sum()}/{tmask3.sum()}")


    if args.mode == 0:
        X1_train = omics_data1.loc[~tmask1, omics_data1.columns[1:]].values
        X2_train = omics_data2.loc[~tmask2, omics_data2.columns[1:]].values
        X3_train = omics_data3.loc[~tmask3, omics_data3.columns[1:]].values

        if X1_train.shape[0] == 0 or X2_train.shape[0] == 0 or X3_train.shape[0] == 0:
            raise ValueError("No train rows left to fit StandardScaler (all in test?) Check your test_list.")

        print("[INFO] Loading Label of train/test data...")
        label_train_df, label_test_df = load_label(args.labeldata, test_ids)
        print(label_train_df.shape, label_train_df, Counter(label_train_df.iloc[:, 1].values))
        print(label_test_df.shape, label_test_df, Counter(label_test_df.iloc[:, 1].values))

        # Train/Val split
        train_idx, val_idx = train_test_split(np.arange(len(label_train_df)), test_size=0.3, random_state=args.seed, stratify=label_train_df['class'])
        label_train_split = label_train_df.iloc[train_idx]
        label_val_split = label_train_df.iloc[val_idx]
        print(Counter(label_val_split.iloc[:, 1].values))
       
        x1_train = omics_data1.iloc[train_idx, 1:].values
        x2_train = omics_data2.iloc[train_idx, 1:].values
        x3_train = omics_data3.iloc[train_idx, 1:].values

        scaler1 = StandardScaler().fit(x1_train)
        scaler2 = StandardScaler().fit(x2_train)
        scaler3 = StandardScaler().fit(x3_train)

        joblib.dump(scaler1, scaler1_path)
        joblib.dump(scaler2, scaler2_path)
        joblib.dump(scaler3, scaler3_path)

        omics_data1.iloc[:, 1:] = scaler1.transform(omics_data1.iloc[:, 1:].values)
        omics_data2.iloc[:, 1:] = scaler2.transform(omics_data2.iloc[:, 1:].values)
        omics_data3.iloc[:, 1:] = scaler3.transform(omics_data3.iloc[:, 1:].values)

        
    elif args.mode == 2:
        print("[INFO] Loading Label of train/test data...")
        label_train_df, label_test_df = load_label(args.labeldata, test_ids)

        train_idx, val_idx = train_test_split(np.arange(len(label_train_df)), test_size=0.3, random_state=args.seed, stratify=label_train_df['class'])
        label_train_split = label_train_df.iloc[train_idx]
        label_val_split = label_train_df.iloc[val_idx]

        if not (os.path.exists(scaler1_path) and os.path.exists(scaler2_path) and os.path.exists(scaler3_path)):
            raise FileNotFoundError("Scaler pkl files not found. Run with mode 0/1 first to create scalers.")
        scaler1 = joblib.load(scaler1_path)
        scaler2 = joblib.load(scaler2_path)
        scaler3 = joblib.load(scaler3_path)

        omics_data1.iloc[:, 1:] = scaler1.transform(omics_data1.iloc[:, 1:].values)
        omics_data2.iloc[:, 1:] = scaler2.transform(omics_data2.iloc[:, 1:].values)
        omics_data3.iloc[:, 1:] = scaler3.transform(omics_data3.iloc[:, 1:].values)

    in_feas = [
        omics_data1.shape[1] - 1,
        omics_data2.shape[1] - 1,
        omics_data3.shape[1] - 1
    ]

    ## Merge ##
    Merge_data = pd.merge(omics_data1, omics_data2, on='Sample', how='inner')
    Merge_data = pd.merge(Merge_data, omics_data3, on='Sample', how='inner')
    Merge_data.sort_values(by='Sample', ascending=True, inplace=True)  # index 맞춰줌

    print('[Train + Test]', Merge_data.shape)

    # Train/Test
    work(Merge_data, label_train_split, label_val_split, label_test_df, \
         in_feas, lr=args.learningrate, bs=args.batchsize, epochs=args.epoch, device=device, a=args.a, b=args.b, c=args.c, mode=args.mode)
    print('Success! Results can be seen in result file')