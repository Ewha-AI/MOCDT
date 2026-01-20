#!/usr/bin/env bash
# Stop at first error
set -e

echo "Starting GCN_run_cv.py"

export CUDA_VISIBLE_DEVICES=2

python GCN_run_cv.py \
  -fd_tr ./result_latent_CV/Contrastive_Supervised_Adversarial_ae_CV/[All_latent]_244_d750_adv0.05_con0.2/latent_train.csv \
  -fd_te ./result_latent_CV/Contrastive_Supervised_Adversarial_ae_CV/[All_latent]_244_d750_adv0.05_con0.2/latent_test.csv \
  -ad_tr ./result_SNF/latent_snf_K15/244_d750/SNF_fused_train.csv  \
  -ad_te ./result_SNF/latent_snf_K15/244_d750/SNF_fused_test.csv  \
  -ld_tr ./data/sample_classes_train.csv \
  -ld_te ./data/sample_classes_test.csv \
  -nc 9 \
  -d gpu \
  -e 10000 \
  -p 2000 \
  -hd 512 \
  -dp 0.1 \
  -t 0.01 \
  -w 0.0001 \
  -f CDTOO_gcn \
  -tspec 0.99 \
  -sf 244_d750_adv0.05_con0.2_thsd0.99_Drop0.1

echo "finished."
