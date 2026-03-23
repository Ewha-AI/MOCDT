# MOCDT: Multi-Cancer Detection and Tissue-of-Origin Classification via cfDNA Multi-modal Integration
Cancer Detection and Tissue-of-Origin prediction framework using multi-omics cfDNA data.
## MOCDT
<img width="4119" height="2033" alt="Model_" src="https://github.com/user-attachments/assets/e03ecaf9-e87f-4822-a3d0-19e7c291ac1f" />

MOCDT is a cell-free DNA (cfDNA) multi-omics framework that follows a clinically aligned two-stage pipeline: high-specificity CD followed by conditional TOO classification. MOCDT combines (i) a supervised multi-modal autoencoder incorporating adversarial modality alignment and supervised contrastive geometry shaping, with (ii) a latent space patient similarity network and (iii) a residual GCN for relational learning. Applied to a cfDNA cohort including healthy controls and eight cancer types, MOCDT achieved 95.74\% specificity and 96.22\% sensitivity for CD at a high-specificity operating point, and 75.2\% Top1 and 91.06\% Top3 accuracy for TOO classification. Latent attribution analysis showed that the model learns tissue-dependent latent features rather than relying on a single universal biomarker axis. Together, these results demonstrate that MOCDT enables accurate and interpretable cfDNA-based multi-omics integration, supporting clinically relevant liquid biopsy applications.

The `sample_data` folder includes five raw samples for each modality, which can be used for model validation. GCN inference can be performed using these data together with the _pretrained model_ available on the [Google drive](https://drive.google.com/drive/folders/1_4G-9qhwgZsm_UMOA2BGuL-4l0MZdWin?usp=drive_link). 

### Step 1. Supervised Multi-modal Autoencoder
The script supports three modes:
- `mode 0`: AE training on the training dataset  
- `mode 1`: Latent representation extraction on the test dataset  
- `mode 2`: Latent representation extraction on the sample dataset

```
./run_AE.sh
```

### Step 2. Latent Space Patient Similarity Network
To get a fused patient similarity graph, you can run the following command:
```
./run_snf_LATENT.sh
```
To generate the patient similarity graph for the sample dataset, simply specify the dataset path using the `-val_p` argument.

### Step 3. Residual Graph Convolutional Network for Classification
To **train, test**, and perform **validation** for CD and TOO, you can run the following command:
```
./run_GCN_CDTOO.sh
```

