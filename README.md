# MOCDT: Multi-Cancer Detection and Tissue-of-Origin Classification via cfDNA Multi-modal Integration
Cancer Detection and Tissue-of-origin prediction framework using multi-omics cfDNA data
## MOCDT
<img width="4079" height="1986" alt="Model" src="https://github.com/user-attachments/assets/42fe370a-56e9-432a-8190-a82831019605" />

MOCDT is a cell-free DNA (cfDNA) multi-omics framework that follows a clinically aligned two-stage pipeline: high-specificity CD followed by conditional TOO classification. MOCDT combines (i) a supervised multi-modal autoencoder incorporating adversarial modality alignment and supervised contrastive geometry shaping, with (ii) a latent space patient similarity network and (iii) a residual GCN for relational learning. Applied to a cfDNA cohort including healthy controls and eight cancer types, MOCDT achieved 95.74\% specificity and 96.22\% sensitivity for CD at a high-specificity operating point, and 75.2\% Top1 and 91.06\% Top3 accuracy for TOO classification. Latent attribution analysis showed that the model learns tissue-dependent latent features rather than relying on a single universal biomarker axis. Together, these results demonstrate that MOCDT enables accurate and interpretable cfDNA-based multi-omics integration, supporting clinically relevant liquid biopsy applications.

### Supervised Multi-modal Autoencoder
```
./run_Contrastive_Supervised_Adversarial_AE.sh
```
### Latent Space Patient Similarity Network
```
./run_snf_LATENT.sh
```
### Residual Graph Convolutional Network for Classification
```
./run_GCN_CDTOO.sh
```
