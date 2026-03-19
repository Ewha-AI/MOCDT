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

    # fused latent
    save_latent_to_dataframe(latent_data, sample_name, save_path + '.csv')
    # per modality
    save_latent_to_dataframe(latent_z1, sample_name, save_path + '_met.csv')
    save_latent_to_dataframe(latent_z2, sample_name, save_path + '_fsr.csv')
    save_latent_to_dataframe(latent_z3, sample_name, save_path + '_cnv.csv')


def work(data=None, external_val_data=None, label_train_df=None, label_val_df=None, label_test_df=None, in_feas=None, lr=0.001, bs=32, epochs=100, device=torch.device('cpu'), a=0.4, b=0.3, c=0.3, mode=0, topn=100):
    model_dir = os.path.join("model_pth/AE", args.filename, args.subfolder)
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pt")
    
    if mode == 2:   # mode 2: external only
        if external_val_data is None:
            raise ValueError("mode 2 requires external_val_data.")

        print('\n[Mode 2] External-only inference')
        print('** External Validation **')
        print(external_val_data)

        X_external_val = external_val_data.iloc[:, 1:].to_numpy()
        print('External Data shape:', X_external_val.shape)

        TX_external_val = torch.tensor(X_external_val, dtype=torch.float, device=device)

        print('\nLoad saved model and generate external latent...')
        mmae = model.Contrastive_Triplet_Supervised_Adversarial_AE_model_cv.Contrastive_SV_Adversarial_MMAE(
            in_feas,
            latent_dim=args.latent,
            num_classes=9,
            a=a, b=b, c=c,
            lambda_adv=args.lambda_adv,
            lambda_con=args.lambda_con
        )
        mmae.load_state_dict(torch.load(best_model_path, map_location=device))
        mmae.to(device)
        mmae.eval()

        out_name = "latent_external_val"
        save_latent(
            mmae,
            TX_external_val,
            external_val_data['Sample'].tolist(),
            in_feas,
            os.path.join(save_dir, out_name)
        )
        return

    if data is None:    # mode 0 / 1: must include main data
        raise ValueError(f"mode {mode} requires main merged data (data).")

    train_data = pd.merge(label_train_df[['Sample_ID']], data, left_on='Sample_ID', right_on='Sample', how='left')
    val_data = pd.merge(label_val_df[['Sample_ID']], data, left_on='Sample_ID', right_on='Sample', how='left')
    test_data = data[data['Sample'].isin(label_test_df['Sample_ID'])]

    train_data = train_data[['Sample'] + [col for col in data.columns if col != 'Sample']]
    val_data = val_data[['Sample'] + [col for col in data.columns if col != 'Sample']]

    # print('\nData-Label Check!')
    # print('** Train split **')
    # print(train_data)
    # print(label_train_df)
    # print((train_data['Sample'].values == label_train_df['Sample_ID'].values).sum())
    # print('** Val split **')
    # print(val_data)
    # print(label_val_df)
    # print((val_data['Sample'].values == label_val_df['Sample_ID'].values).sum())
    # print('** Test **')
    # print(test_data)
    # print(label_test_df)
    # print(len(set(test_data['Sample']) & set(label_test_df['Sample_ID'])))
    # if external_val_data is not None:
    #     print('** External Validation **')
    #     print(external_val_data)

    X_train = train_data.iloc[:, 1:].to_numpy()
    Y_train = label_train_df.iloc[:, 1].to_numpy()
    X_val = val_data.iloc[:, 1:].to_numpy()
    Y_val = label_val_df.iloc[:, 1].to_numpy()

    X_test = test_data.iloc[:, 1:].to_numpy()

    X_external_val = None
    if external_val_data is not None:
        X_external_val = external_val_data.iloc[:, 1:].to_numpy()
        print(f'External Min: {X_external_val.min()}  Max: {X_external_val.max()}')
        trainval_data = pd.concat([train_data, val_data])
        print(trainval_data)

        X_external_val = np.clip(
            X_external_val,
            trainval_data.iloc[:, 1:].values.min(),
            trainval_data.iloc[:, 1:].values.max()
        )
        print(f'[Remove Outlier] Min: {X_external_val.min()}  Max: {X_external_val.max()}')

    print('Data shape:',
          X_train.shape, X_val.shape, X_test.shape,
          X_external_val.shape if X_external_val is not None else None)

    print(Counter(Y_val))

    TX_train = torch.tensor(X_train, dtype=torch.float, device=device)
    TY_train = torch.tensor(Y_train, dtype=torch.long, device=device)
    TX_val = torch.tensor(X_val, dtype=torch.float, device=device)
    TY_val = torch.tensor(Y_val, dtype=torch.long, device=device)

    TX_test = torch.tensor(X_test, dtype=torch.float, device=device)
    TX_external_val = None
    if X_external_val is not None:
        TX_external_val = torch.tensor(X_external_val, dtype=torch.float, device=device)

    # Train
    if mode == 0:
        print('\nTraining model...')
        train_Tensor_data = Data.TensorDataset(TX_train, TY_train)
        val_Tensor_data = Data.TensorDataset(TX_val, TY_val)
        train_loader = Data.DataLoader(train_Tensor_data, batch_size=bs, shuffle=True)
        val_loader = Data.DataLoader(val_Tensor_data, batch_size=bs, shuffle=True)

        mmae = model.Contrastive_Triplet_Supervised_Adversarial_AE_model_cv.Contrastive_SV_Adversarial_MMAE(
            in_feas,
            latent_dim=args.latent,
            num_classes=9,
            a=a, b=b, c=c,
            lambda_adv=args.lambda_adv,
            lambda_con=args.lambda_con
        )
        mmae.to(device)
        mmae.train()
        mmae.train_MMAE(model_dir, save_dir, train_loader, val_loader,
                        learning_rate=lr, device=device, epochs=epochs)
        mmae.eval()

    # Load saved model & Inference
    if mode in [0, 1]:
        print('\nGet the latent layer output...')
        mmae = model.Contrastive_Triplet_Supervised_Adversarial_AE_model_cv.Contrastive_SV_Adversarial_MMAE(
            in_feas,
            latent_dim=args.latent,
            num_classes=9,
            a=a, b=b, c=c,
            lambda_adv=args.lambda_adv,
            lambda_con=args.lambda_con
        )
        mmae.load_state_dict(torch.load(best_model_path, map_location=device))
        mmae.to(device)
        mmae.eval()

        if mode == 0:
            out_name = "latent_train"
            TX_Train = torch.cat([TX_train, TX_val], dim=0)
            save_latent(
                mmae,
                TX_Train,
                train_data['Sample'].tolist() + val_data['Sample'].tolist(),
                in_feas,
                os.path.join(save_dir, out_name)
            )

        elif mode == 1:
            print(f"\n[INFO] Generating latent for Test set...")
            out_name = "latent_test"
            save_latent(
                mmae,
                TX_test,
                test_data['Sample'].tolist(),
                in_feas,
                os.path.join(save_dir, out_name)
            )

            if TX_external_val is not None:
                print(f"\n[INFO] Generating latent for External Validation set...")
                out_name = "latent_external_val"
                save_latent(
                    mmae,
                    TX_external_val,
                    external_val_data['Sample'].tolist(),
                    in_feas,
                    os.path.join(save_dir, out_name)
                )

    return


def validate_args(args):
    has_main = all([args.path1, args.path2, args.path3])
    has_any_main = any([args.path1, args.path2, args.path3])

    has_ext = all([args.ext_path1, args.ext_path2, args.ext_path3])
    has_any_ext = any([args.ext_path1, args.ext_path2, args.ext_path3])

    if has_any_main and not has_main:
        raise ValueError("If using main omics data, you must provide all of path1/path2/path3.")

    if has_any_ext and not has_ext:
        raise ValueError("If using external omics data, you must provide all of ext_path1/ext_path2/ext_path3.")

    if args.mode in [0, 1]:
        if not has_main:
            raise ValueError(f"mode {args.mode} requires path1/path2/path3.")

    if args.mode == 2:
        if not has_ext:
            raise ValueError("mode 2 requires ext_path1/ext_path2/ext_path3.")

    if args.mode in [0, 1] and args.labeldata is None:
        raise ValueError(f"mode {args.mode} requires --labeldata.")

    return has_main, has_ext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=int, choices=[0, 1, 2], default=0,
                        help='Mode 0: train&integrate, '
                             'Mode 1: infer main(test) + optional external, '
                             'Mode 2: infer external only.')

    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')

    # main omics
    parser.add_argument('--path1', '-p1', type=str, default=None, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', type=str, default=None, help='The second omics file name.')
    parser.add_argument('--path3', '-p3', type=str, default=None, help='The third omics file name.')

    # external omics
    parser.add_argument('--ext_path1', '-extp1', type=str, default=None, help='The first omics file name (External Validation).')
    parser.add_argument('--ext_path2', '-extp2', type=str, default=None, help='The second omics file name (External Validation).')
    parser.add_argument('--ext_path3', '-extp3', type=str, default=None, help='The third omics file name (External Validation).')

    parser.add_argument('--batchsize', '-bs', type=int, default=32, help='Training batchsize, default: 32.')
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
    parser.add_argument('--test_list', '-tl', type=str, default='data/test_sample.csv',
                        help='CSV with test sample IDs; columns: Sample_ID or Sample')
    parser.add_argument('--tag', '-t', type=str, default=None, help='Suffix for latent csv filename, e.g., train or test')

    parser.add_argument('--labeldata', '-ld', type=str, default=None)
    parser.add_argument('--lambda_cls', '-lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_adv', '-lambda_adv', type=float, required=True)
    parser.add_argument('--lambda_con', '-lambda_con', type=float, required=True)

    args = parser.parse_args()

    has_main, has_ext = validate_args(args)

    save_dir = os.path.join("result_latent_CV", args.filename, args.subfolder)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    scaler1_path = os.path.join(save_dir, "scaler_omics1.pkl")
    scaler2_path = os.path.join(save_dir, "scaler_omics2.pkl")
    scaler3_path = os.path.join(save_dir, "scaler_omics3.pkl")

    log_mode_dict = {
        0: 'Train',
        1: 'Test_Externalval',
        2: 'ExternalOnly'
    }
    log_path = os.path.join(save_dir, f"log_{log_mode_dict.get(args.mode, 'Run')}.txt")
    stdoutOrigin = sys.stdout
    sys.stdout = Logger(log_path)

    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_seed(args.seed)

    if args.a + args.b + args.c != 1.0:
        print('The sum of weights must be 1.')
        exit(1)

    omics_data1, omics_data2, omics_data3 = None, None, None
    external_omics_data1, external_omics_data2, external_omics_data3 = None, None, None

    if has_main:
        omics_data1 = pd.read_csv(args.path1, header=0, index_col=None)
        omics_data2 = pd.read_csv(args.path2, header=0, index_col=None)
        omics_data3 = pd.read_csv(args.path3, header=0, index_col=None)

        omics_data1.rename(columns={omics_data1.columns.tolist()[0]: 'Sample'}, inplace=True)
        omics_data2.rename(columns={omics_data2.columns.tolist()[0]: 'Sample'}, inplace=True)
        omics_data3.rename(columns={omics_data3.columns.tolist()[0]: 'Sample'}, inplace=True)

        omics_data1.sort_values(by='Sample', ascending=True, inplace=True)
        omics_data2.sort_values(by='Sample', ascending=True, inplace=True)
        omics_data3.sort_values(by='Sample', ascending=True, inplace=True)

    if has_ext:
        external_omics_data1 = pd.read_csv(args.ext_path1, header=0, index_col=None)
        external_omics_data2 = pd.read_csv(args.ext_path2, header=0, index_col=None, sep='\t')
        external_omics_data3 = pd.read_csv(args.ext_path3, header=0, index_col=None, sep='\t')
        print(external_omics_data1.shape, external_omics_data2.shape, external_omics_data3.shape)

        external_omics_data1.rename(columns={external_omics_data1.columns.tolist()[0]: 'Sample'}, inplace=True)
        external_omics_data2.rename(columns={external_omics_data2.columns.tolist()[0]: 'Sample'}, inplace=True)
        external_omics_data3.rename(columns={external_omics_data3.columns.tolist()[0]: 'Sample'}, inplace=True)

        external_omics_data1.sort_values(by='Sample', ascending=True, inplace=True)
        external_omics_data2.sort_values(by='Sample', ascending=True, inplace=True)
        external_omics_data3.sort_values(by='Sample', ascending=True, inplace=True)


    if has_main:
        in_feas = [
            omics_data1.shape[1] - 1,
            omics_data2.shape[1] - 1,
            omics_data3.shape[1] - 1
        ]
    elif has_ext:
        in_feas = [
            external_omics_data1.shape[1] - 1,
            external_omics_data2.shape[1] - 1,
            external_omics_data3.shape[1] - 1
        ]
    else:
        raise ValueError("No valid input data provided.")


    label_train_split, label_val_split, label_test_df = None, None, None
    Merge_data, external_Merge_data = None, None

    print("mode =", args.mode)

    if args.mode in [0, 1]:
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

        print('Test ID:', len(test_ids), list(test_ids)[:5], '...')
        print(f"[TEST IDs] in omics1/2/3: {tmask1.sum()}/{tmask2.sum()}/{tmask3.sum()}")

        print("[INFO] Loading Label of train/test data...")
        label_train_df, label_test_df = load_label(args.labeldata, test_ids)

        print(label_train_df.shape, label_train_df, Counter(label_train_df.iloc[:, 1].values))
        print(label_test_df.shape, label_test_df, Counter(label_test_df.iloc[:, 1].values))

        train_idx, val_idx = train_test_split(
            np.arange(len(label_train_df)),
            test_size=0.3,
            random_state=args.seed,
            stratify=label_train_df['class']
        )
        label_train_split = label_train_df.iloc[train_idx]
        label_val_split = label_train_df.iloc[val_idx]
        print(Counter(label_val_split.iloc[:, 1].values))

        if args.mode == 0:
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

            if has_ext:
                external_omics1_features = external_omics_data1.iloc[:, 1:].astype('float64')
                external_omics_data1 = pd.concat([external_omics_data1[['Sample']], external_omics1_features], axis=1)
                # external_omics_data1.iloc[:, 1:] = scaler1.transform(external_omics_data1.iloc[:, 1:].values)
                external_omics_data2.iloc[:, 1:] = scaler2.transform(external_omics_data2.iloc[:, 1:].values)
                external_omics_data3.iloc[:, 1:] = scaler3.transform(external_omics_data3.iloc[:, 1:].values)

        elif args.mode == 1:
            if not (os.path.exists(scaler1_path) and os.path.exists(scaler2_path) and os.path.exists(scaler3_path)):
                raise FileNotFoundError("Scaler pkl files not found. Run with mode 0 first to create scalers.")

            scaler1 = joblib.load(scaler1_path)
            scaler2 = joblib.load(scaler2_path)
            scaler3 = joblib.load(scaler3_path)

            omics_data1.iloc[:, 1:] = scaler1.transform(omics_data1.iloc[:, 1:].values)
            omics_data2.iloc[:, 1:] = scaler2.transform(omics_data2.iloc[:, 1:].values)
            omics_data3.iloc[:, 1:] = scaler3.transform(omics_data3.iloc[:, 1:].values)

            if has_ext:
                external_omics1_features = external_omics_data1.iloc[:, 1:].astype('float64')
                external_omics_data1 = pd.concat([external_omics_data1[['Sample']], external_omics1_features], axis=1)
                # external_omics_data1.iloc[:, 1:] = scaler1.transform(external_omics_data1.iloc[:, 1:].values)
                external_omics_data2.iloc[:, 1:] = scaler2.transform(external_omics_data2.iloc[:, 1:].values)
                external_omics_data3.iloc[:, 1:] = scaler3.transform(external_omics_data3.iloc[:, 1:].values)

        Merge_data = pd.merge(omics_data1, omics_data2, on='Sample', how='inner')
        Merge_data = pd.merge(Merge_data, omics_data3, on='Sample', how='inner')
        Merge_data.sort_values(by='Sample', ascending=True, inplace=True)
        print('[Train + Test]', Merge_data.shape)

        if has_ext:
            external_Merge_data = pd.merge(external_omics_data1, external_omics_data2, on='Sample', how='inner')
            external_Merge_data = pd.merge(external_Merge_data, external_omics_data3, on='Sample', how='inner')
            external_Merge_data.sort_values(by='Sample', ascending=True, inplace=True)
            print('[External Val]', external_Merge_data.shape)

        work(
            data=Merge_data, external_val_data=external_Merge_data, label_train_df=label_train_split, label_val_df=label_val_split, label_test_df=label_test_df, in_feas=in_feas, lr=args.learningrate, bs=args.batchsize, epochs=args.epoch, device=device, a=args.a, b=args.b, c=args.c, mode=args.mode)

    # mode 2: external-only
    elif args.mode == 2:
        if not (os.path.exists(scaler1_path) and os.path.exists(scaler2_path) and os.path.exists(scaler3_path)):
            raise FileNotFoundError("Scaler pkl files not found. Run with mode 0 first to create scalers.")

        scaler1 = joblib.load(scaler1_path)
        scaler2 = joblib.load(scaler2_path)
        scaler3 = joblib.load(scaler3_path)

        external_omics1_features = external_omics_data1.iloc[:, 1:].astype('float64')
        external_omics_data1 = pd.concat([external_omics_data1[['Sample']], external_omics1_features], axis=1)
        external_omics_data2.iloc[:, 1:] = scaler2.transform(external_omics_data2.iloc[:, 1:].values)
        external_omics_data3.iloc[:, 1:] = scaler3.transform(external_omics_data3.iloc[:, 1:].values)

        external_Merge_data = pd.merge(external_omics_data1, external_omics_data2, on='Sample', how='inner')
        external_Merge_data = pd.merge(external_Merge_data, external_omics_data3, on='Sample', how='inner')
        external_Merge_data.sort_values(by='Sample', ascending=True, inplace=True)

        print('[External Only]', external_Merge_data.shape)

        work(data=None, external_val_data=external_Merge_data, label_train_df=None, label_val_df=None, label_test_df=None, in_feas=in_feas, lr=args.learningrate, bs=args.batchsize, epochs=args.epoch, device=device, a=args.a, b=args.b, c=args.c, mode=args.mode)

    print('Success! Results can be seen in result file')