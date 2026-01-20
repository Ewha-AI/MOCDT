#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from matplotlib import pyplot as plt
import os
import numpy as np

def make_triplets(latent, labels, max_triplets_per_class=10):
    """
    latent: [N, latent_dim]
    labels: [N]
    return:
      anchors, positives, negatives : [M, latent_dim]
    """
    device = latent.device
    labels = labels.cpu().numpy()
    unique_labels = np.unique(labels)

    anchor_list = []
    pos_list = []
    neg_list = []

    for cls in unique_labels:
        idx_pos = np.where(labels == cls)[0]
        idx_neg = np.where(labels != cls)[0]

        if len(idx_pos) < 2 or len(idx_neg) < 1:
            continue  # triplet 못 만드는 클래스는 스킵

        # 최대 max_triplets_per_class 개만 사용 (너무 많으면 학습 불안정해질 수 있어서)
        num_triplets = min(max_triplets_per_class, len(idx_pos) * (len(idx_pos) - 1))

        for _ in range(num_triplets):
            i_anchor, i_pos = np.random.choice(idx_pos, size=2, replace=False)
            i_neg = np.random.choice(idx_neg)

            anchor_list.append(latent[i_anchor])
            pos_list.append(latent[i_pos])
            neg_list.append(latent[i_neg])

    if len(anchor_list) == 0:
        return None, None, None

    anchors = torch.stack(anchor_list).to(device)
    positives = torch.stack(pos_list).to(device)
    negatives = torch.stack(neg_list).to(device)
    return anchors, positives, negatives


class Contrastive_SV_Adversarial_MMAE(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, num_classes, a=0.4, b=0.3, c=0.3,
                 lambda_adv=0.1, lambda_rec=1.0, lambda_cls=1.0, lambda_con=0.1):
        '''
        :param in_feas_dim: a list, input dims of omics data
        :param latent_dim: dim of latent layer
        :param a: weight of omics data type 1
        :param b: weight of omics data type 2
        :param c: weight of omics data type 3
        '''
        super(Contrastive_SV_Adversarial_MMAE, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.in_feas = in_feas_dim
        self.latent = latent_dim
        self.num_classes = num_classes
        self.lambda_adv = lambda_adv
        self.lambda_rec = lambda_rec
        self.lambda_cls = lambda_cls
        self.lambda_con = lambda_con  ##

        # Encoders (AE)
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(self.in_feas[0], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(self.in_feas[1], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        self.encoder_omics_3 = nn.Sequential(
            nn.Linear(self.in_feas[2], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid()
        )
        # Decoders  (AE)
        self.decoder_omics_1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        self.decoder_omics_2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        self.decoder_omics_3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))

        # Classifier head (Supervised)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,num_classes) # logits
        )

        # Discriminator (Modality Alignment): 모달리티 구분
        self.discriminator = nn.Sequential(
            nn.Linear(self.latent, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 0: omics_1, 1: omics_2, 2: omics_3
        )

        # Weight initialization
        for name, param in Contrastive_SV_Adversarial_MMAE.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def encode_each(self, omics_1, omics_2, omics_3):
        """모달리티별 latent (공유 공간)만 따로 뽑는 헬퍼 함수."""
        z1 = self.encoder_omics_1(omics_1)
        z2 = self.encoder_omics_2(omics_2)
        z3 = self.encoder_omics_3(omics_3)
        return z1, z2, z3

    def fuse_latent(self, z1, z2, z3):
        """고정 가중치 a,b,c로 모달리티 latent를 합치는 부분."""
        return self.a * z1 + self.b * z2 + self.c * z3

    def decode_all(self, latent_data):
        """하나의 fused latent에서 세 모달리티 모두 재구성."""
        x1_hat = self.decoder_omics_1(latent_data)
        x2_hat = self.decoder_omics_2(latent_data)
        x3_hat = self.decoder_omics_3(latent_data)
        return x1_hat, x2_hat, x3_hat
    

    def forward(self, omics_1, omics_2, omics_3):
        '''
        :param omics_1: omics data 1
        :param omics_2: omics data 2
        :param omics_3: omics data 3
        :return:
            latent_data          : [N, latent_dim]
            decoded_omics_1..3   : reconstructions
            logits               : [N, num_classes]
        '''
        z1, z2, z3 = self.encode_each(omics_1, omics_2, omics_3)
        latent = self.fuse_latent(z1, z2, z3)
        x1_hat, x2_hat, x3_hat = self.decode_all(latent)
        logits = self.classifier(latent)
        return latent, x1_hat, x2_hat, x3_hat, logits, z1, z2, z3


    def train_MMAE(self, model_dir, save_dir, train_loader, val_loader, learning_rate=0.001, device=torch.device('cpu'), \
                   epochs=100):
        
        # 파라미터 분리: AE+Classifier vs Discriminator
        ae_params = (
            list(self.encoder_omics_1.parameters()) +
            list(self.encoder_omics_2.parameters()) +
            list(self.encoder_omics_3.parameters()) +
            list(self.decoder_omics_1.parameters()) +
            list(self.decoder_omics_2.parameters()) +
            list(self.decoder_omics_3.parameters()) +
            list(self.classifier.parameters())
        )
        D_params = list(self.discriminator.parameters())

        # Optimizer
        optimizer_ae = torch.optim.Adam(ae_params, lr=learning_rate)
        optimizer_D  = torch.optim.Adam(D_params,  lr=learning_rate)

        # Loss functions
        rec_loss_fn = nn.MSELoss()
        adv_loss_fn = nn.CrossEntropyLoss()
        cls_loss_fn = nn.CrossEntropyLoss()   # multi-class (num_classes)
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)


        train_loss_ls, val_loss_ls = [],[]
        best_val_loss = float('inf')
        f0,f1,f2 = self.in_feas
        self.to(device)

        for epoch in range(epochs):
            ## Train ##
            self.train()
            train_loss_sum = 0.0   
            correct_train = 0
            total_train = 0   # label 전체
            for (x, y) in train_loader:
                omics_1 = x[:, :f0].to(device)
                omics_2 = x[:, f0: f0+f1].to(device)
                omics_3 = x[:, f0+f1 : f0+f1+f2].to(device)
                N = omics_1.size(0)
                y = y.to(device).long()

                # Update Discriminator
                with torch.no_grad():
                    z1, z2, z3 = self.encode_each(omics_1, omics_2, omics_3)
                Z_all = torch.cat([z1, z2, z3], dim=0)  # [3N, latent]
                dom_labels = torch.cat([
                    torch.zeros(N, dtype=torch.long),
                    torch.ones(N, dtype=torch.long),
                    2 * torch.ones(N, dtype=torch.long)
                ], dim=0).to(device)  # [3N]: 각 모달리티별 latent (0/1/2)

                logits_D = self.discriminator(Z_all.detach())
                loss_D = adv_loss_fn(logits_D, dom_labels)

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                # Update AE + Classifier
                z1, z2, z3 = self.encode_each(omics_1, omics_2, omics_3)
                latent = self.fuse_latent(z1, z2, z3)
                x1_hat, x2_hat, x3_hat = self.decode_all(latent)
                logits_cls = self.classifier(latent)

                ## Reconstruction
                L_rec = (
                    self.a * rec_loss_fn(x1_hat, omics_1) +
                    self.b * rec_loss_fn(x2_hat, omics_2) +
                    self.c * rec_loss_fn(x3_hat, omics_3)
                )

                ## Classification
                L_cls = cls_loss_fn(logits_cls, y)

                ## Adversarial  (Encoder에서 최대화: domain confusion)
                Z_enc = torch.cat([z1, z2, z3], dim=0)
                logits_dom_enc = self.discriminator(Z_enc)
                L_adv = adv_loss_fn(logits_dom_enc, dom_labels)


                ## Contrastive (Triplet)
                anchors, positives, negatives = make_triplets(latent, y, max_triplets_per_class=10)
                if anchors is not None:
                    L_triplet = triplet_loss_fn(anchors, positives, negatives)
                else:
                    L_triplet = torch.tensor(0.0, device=device)

                total_loss = (
                    self.lambda_rec * L_rec +
                    self.lambda_cls * L_cls -
                    self.lambda_adv * L_adv +
                    self.lambda_con * L_triplet  ##
                )

                optimizer_ae.zero_grad()
                total_loss.backward()
                optimizer_ae.step()

                train_loss_sum += total_loss.item()*x.size(0)

                # train accuracy
                with torch.no_grad():
                    pred = logits_cls.argmax(dim=1)
                    correct_train += (pred==y).sum().item()
                    total_train += y.size(0)

            avg_train_loss = train_loss_sum / max(1, total_train)
            train_acc = correct_train / max(1, total_train)
            train_loss_ls.append(avg_train_loss)

            ## Validation ##
            self.eval()
            val_loss_sum = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for (x_val, y_val) in val_loader:
                    omics_1 = x_val[:, :f0].to(device)
                    omics_2 = x_val[:, f0:f0+f1].to(device)
                    omics_3 = x_val[:, f0+f1:f0+f1+f2].to(device)
                    y_val = y_val.to(device).long()

                    z1_v, z2_v, z3_v = self.encode_each(omics_1, omics_2, omics_3)
                    latent_v = self.fuse_latent(z1_v, z2_v, z3_v)
                    x1_hat_v, x2_hat_v, x3_hat_v = self.decode_all(latent_v)
                    logits_v = self.classifier(latent_v)

                    L_rec_v = (
                        self.a * rec_loss_fn(x1_hat_v, omics_1) +
                        self.b * rec_loss_fn(x2_hat_v, omics_2) +
                        self.c * rec_loss_fn(x3_hat_v, omics_3)
                    )
                    L_cls_v = cls_loss_fn(logits_v, y_val)
                    val_loss = self.lambda_rec * L_rec_v + self.lambda_cls * L_cls_v

                    val_loss_sum += val_loss.item() * x_val.size(0)

                    pred_v = logits_v.argmax(dim=1)
                    correct_val += (pred_v == y_val).sum().item()
                    total_val += y_val.size(0)

            avg_val_loss = val_loss_sum / max(1, total_val)
            val_acc = correct_val / max(1, total_val)
            val_loss_ls.append(avg_val_loss)

            print(f"[Epoch {epoch+1:03d}] "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} || Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # best model 저장 (val loss 기준)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print('** Save best model (val loss) **')
                torch.save(self.state_dict(), os.path.join(model_dir, "best_model.pt"))

        # 마지막 epoch 모델 저장
        print('** Save last model **')
        torch.save(self.state_dict(), os.path.join(model_dir, "last_model.pt"))
        
        # plot training loss
        plt.figure()
        plt.plot(range(1, epochs+1), train_loss_ls, label='Train Loss')
        plt.plot(range(1, epochs+1), val_loss_ls, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Adversarial Supervised AE Loss (Adv + Recon + Class)')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "AE_Adversarial_Supervised_train_val_loss.png"))
        plt.close()