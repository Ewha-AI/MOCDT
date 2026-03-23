#!/usr/bin/env bash
set -e

echo "Starting GCN_inference.py"

export CUDA_VISIBLE_DEVICES=0

python GCN_inference.py \
  -fd ./result_latent_CV/Contrastive_Supervised_Adversarial_ae_CV/[All_latent]_244_d750_adv0.05_con0.2/latent_external_val.csv \
  -ad ./result/latent_snf_K15/244_d750/SNF_fused_val.csv \
  -sl ./data/validation_sample.csv \
  -ld ./data/sample_classes_merged.csv \
  --binary_model_path ./model_pth/GCN/CDTOO_gcn/244_d750_adv0.05_con0.2_thsd0.99_Drop0.1/best_binary_model.pkl \
  --too_model_path ./model_pth/GCN/CDTOO_gcn/244_d750_adv0.05_con0.2_thsd0.99_Drop0.1/best_too_model.pkl \
  --binary_threshold 0.99 \
  -nc 9 \
  -hd 512 \
  -dp 0.1 \
  -d gpu \
  -f GCN_inference \
  -sf 244_d750_adv0.05_con0.2

echo "finished."