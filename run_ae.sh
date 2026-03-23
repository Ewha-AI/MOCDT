#!/usr/bin/env bash
# Stop at first error
set -e

echo "Starting AE_run.py"

export CUDA_VISIBLE_DEVICES=0


# mode 0 and mode 1 support execution with extp inputs (optional)
python AE_run.py \
  -p1 ./data/met/merged_110k.csv \
  -p2 ./data/fsr/Merged_df.csv \
  -p3 ./data/cnv/Merged_df.csv \
  -tl ./data/test_sample.csv \
  -e 100 \
  -l 750 \
  -m 0 \
  -s 0 \
  -d gpu \
  -a 0.2 \
  -b 0.4 \
  -c 0.4 \
  -lambda_cls 1 \
  -lambda_adv 0.05 \
  -lambda_con 0.2 \
  -f Contrastive_Supervised_Adversarial_ae_CV \
  -sf [All_latent]_244_d750_adv0.05_con0.2 \
  -ld ./data/sample_classes_merged.csv

python AE_run.py \
  -p1 ./data/met/merged_110k.csv \
  -p2 ./data/fsr/Merged_df.csv \
  -p3 ./data/cnv/Merged_df.csv \
  -tl ./data/test_sample.csv \
  -e 100 \
  -l 750 \
  -m 1 \
  -s 0 \
  -d gpu \
  -a 0.2 \
  -b 0.4 \
  -c 0.4 \
  -lambda_cls 1 \
  -lambda_adv 0.05 \
  -lambda_con 0.2 \
  -f Contrastive_Supervised_Adversarial_ae_CV \
  -sf [All_latent]_244_d750_adv0.05_con0.2 \
  -ld ./data/sample_classes_merged.csv


# mode 2 uses only extp inputs
python AE_run.py \
  -extp1 ./inference_data/met/Inference_Samples.csv \
  -extp2 ./inference_data/fsr/Inference_Samples.csv \
  -extp3 ./inference_data/cnv/Inference_Samples.csv \
  -e 100 \
  -l 750 \
  -m 2 \
  -s 0 \
  -d gpu \
  -a 0.2 \
  -b 0.4 \
  -c 0.4 \
  -lambda_cls 1 \
  -lambda_adv 0.05 \
  -lambda_con 0.2 \
  -f Contrastive_Supervised_Adversarial_ae_CV \
  -sf [All_latent]_244_d750_adv0.05_con0.2 \
  -ld ./data/sample_classes_merged.csv

echo "finished."

