#!/usr/bin/env bash
# Stop at first error

set -e

echo "Starting SNF.py"

export CUDA_VISIBLE_DEVICES=2

python SNF_latent_total.py \
  -mp ./result_latent_CV/Contrastive_Supervised_Adversarial_ae_CV/[All_latent]_244_d750_adv0.05_con0.2 \
  -train_p latent_train_met.csv latent_train_fsr.csv latent_train_cnv.csv \
  -test_p latent_test_met.csv latent_test_fsr.csv latent_test_cnv.csv \
  -tl ./data/test_sample.csv \
  -k 15 \
  -m sqeuclidean  \
  -f latent_snf_K15 \
  -sf 244_d750

echo "finished."