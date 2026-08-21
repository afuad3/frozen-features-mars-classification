# Frozen Foundation-Model Features for Mars Image Classification

## Research Question

> Can frozen modern vision foundation-model representations provide competitive classification of Martian imagery compared with conventional pretrained CNN representations?

## Hypotheses

- H0: A frozen foundation representation is no better than a frozen ImageNet ResNet-50 representation (Macro-F1) under an identical linear-probe protocol.
- H1: At least one frozen foundation representation (DINOv2, CLIP) provides higher Macro-F1 than frozen ResNet-50.
- Tested per dataset with a paired bootstrap; no direction assumed.

## Datasets

- **HiRISE** (orbital), Zenodo 4002935 v3.2 - 8 classes; **original landmarks only** (10,815) using the released **source-image-grouped** train/val/test split (6,997/2,025/1,793).
- **MSL** (rover), Zenodo 4033453 **v2.1** - 19 classes; official **sol-based** split. MSL v1 (1049137, 24 classes) is documented-only, excluded from the primary comparison.

## Dataset Audit

See `results/audits/data_audit_hirise.md`, `results/audits/data_audit_msl.md` and CSVs.

## Leakage Audit

See `results/audits/leakage_audit_hirise.md`, `results/audits/leakage_audit_msl.md`. HiRISE: assert no source-image / landmark / exact-hash overlap across splits. MSL: single-split assignment, chronological sol ordering, no cross-split hash duplicates. The pipeline STOPS on any failure.

## Historical Baseline

Wagstaff et al. 2021 used **AlexNet fine-tuned** (not a frozen probe). HiRISE 92.8% acc (96.7% @0.9-conf, 20% abstention; most-common 81.1%). MSL 74.5% acc (90.3% calibrated @0.9-conf, 51.8% abstention; most-common 31.2%). These are **context, not a head-to-head** (§21).

## Models

| Representation | Checkpoint | Dim | Feature |
|---|---|---|---|
| ResNet-50 | torchvision IMAGENET1K_V1 | 2048 | global-avg-pool |
| DINOv2 | facebook/dinov2-base | 768 | CLS |
| CLIP | openai/clip-vit-base-patch16 | 512 | image_embeds |

A domain-specific 4th model is documented (candidates: remote-sensing / Mars-specific FMs) and deferred pending checkpoint/license verification.

## Preprocessing

Model-specific official transforms; deterministic (no random augmentation). Grayscale→3-channel replication (no colorization). See `configs/preprocessing.yaml`.

## Experimental Protocol

Frozen embedding → StandardScaler(fit on train) → multinomial L2 logistic regression. (C, class_weight) selected by **validation Macro-F1 only**; final model trained on **train only**; **test evaluated once**. Identical for every representation.

## Evaluation Metrics

Primary: **Macro-F1**. Secondary: balanced accuracy, accuracy, weighted-F1, macro P/R, per-class P/R/F1. 95% bootstrap CIs (paired across models).

## Main Results

| Dataset   | Model    |   Embedding dimension | Classifier                 |   Accuracy |   Balanced Accuracy |   Macro-F1 |   Weighted-F1 |   Macro Precision |   Macro Recall |
|:----------|:---------|----------------------:|:---------------------------|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|
| HIRISE    | resnet50 |                  2048 | Logistic (multinomial, L2) |     0.879  |              0.6832 |     0.6083 |        0.8759 |            0.5626 |         0.6832 |
| HIRISE    | dinov2   |                   768 | Logistic (multinomial, L2) |     0.8974 |              0.8344 |     0.7137 |        0.9042 |            0.6858 |         0.8344 |
| HIRISE    | clip     |                   512 | Logistic (multinomial, L2) |     0.9147 |              0.8278 |     0.7367 |        0.9193 |            0.703  |         0.8278 |
| MSL       | resnet50 |                  2048 | Logistic (multinomial, L2) |     0.7433 |              0.5999 |     0.558  |        0.7381 |            0.5987 |         0.5999 |
| MSL       | dinov2   |                   768 | Logistic (multinomial, L2) |     0.795  |              0.6276 |     0.5977 |        0.7824 |            0.6349 |         0.6276 |
| MSL       | clip     |                   512 | Logistic (multinomial, L2) |     0.7617 |              0.6326 |     0.595  |        0.7652 |            0.6132 |         0.6326 |

## Per-Class Results

See `results/tables/{dataset}_{model}_perclass.csv` and `results/tables/{dataset}_perclass_f1_by_model.csv` (Figs 7–8).

## Error Analysis

See `results/tables/{dataset}_confusion_pairs.csv`, `{dataset}_representative_errors.csv`, and the confusion figures (5–6).

## Representation Analysis

Nearest-neighbor retrieval (cosine on L2 features):

HiRISE:

| model    |   top1_agreement |   top5_agreement |   retrieval_purity |
|:---------|-----------------:|-----------------:|-------------------:|
| resnet50 |         0.863915 |         0.940323 |           0.861461 |
| dinov2   |         0.887897 |         0.951478 |           0.881874 |
| clip     |         0.879532 |         0.953709 |           0.876854 |

MSL:

| model    |   top1_agreement |   top5_agreement |   retrieval_purity |
|:---------|-----------------:|-----------------:|-------------------:|
| resnet50 |         0.713333 |         0.878333 |           0.682667 |
| dinov2   |         0.748333 |         0.898333 |           0.711    |
| clip     |         0.708333 |         0.878333 |           0.683    |

PCA/UMAP scatter: Figs 9–10 (explanatory only). NN panel: Fig 11.

## Statistical Uncertainty

See per-model CIs in the metrics JSONs and `results/tables/{dataset}_paired_bootstrap.csv`. No superiority claimed from non-significant differences.

## Comparison with Wagstaff et al.

Our frozen linear probe is **not** directly comparable to the paper's fine-tuned AlexNet: different model, training regime, preprocessing, and (for MSL) exact split. The 96.7%/90.3% figures include abstention and must not be compared to 100%-coverage accuracy.

## Findings

### HIRISE
- Best frozen representation (Macro-F1): **clip** (Macro-F1=0.7367).
- DINOv2 vs ResNet-50: Δmacro-F1=+0.1054, significant (p=0.000).
- CLIP vs ResNet-50: Δmacro-F1=+0.1284, significant (p=0.000).
- Historical context: Wagstaff AlexNet fine-tuned accuracy=0.928 (100% coverage); our best frozen probe accuracy=0.915. NOT directly comparable (different model/training/preprocessing).

### MSL
- Best frozen representation (Macro-F1): **dinov2** (Macro-F1=0.5977).
- DINOv2 vs ResNet-50: Δmacro-F1=+0.0398, NOT significant (p=0.108).
- CLIP vs ResNet-50: Δmacro-F1=+0.0370, NOT significant (p=0.327).
- Historical context: Wagstaff AlexNet fine-tuned accuracy=0.745 (100% coverage); our best frozen probe accuracy=0.795. NOT directly comparable (different model/training/preprocessing).


## Interpretation - RQ1 questions (§29)

**1. Does a frozen foundation representation contain useful information?** Yes. Macro-F1 far exceeds the majority-class baseline on both datasets (HiRISE base acc 0.827, MSL 0.312); e.g. HiRISE CLIP Macro-F1=0.737, MSL DINOv2 Macro-F1=0.598.
**2. Does DINOv2 outperform ResNet-50?** HiRISE: yes, +0.105 (significant). MSL: +0.040 (NOT significant).
**3. Does CLIP outperform ResNet-50?** HiRISE: yes, +0.128 (significant). MSL: +0.037 (NOT significant).
**4. Does any domain-specific model outperform generic models?** Not evaluated here (domain-specific FM documented and deferred; not a hard dependency).
**5. Are results consistent between HiRISE and MSL?** Directionally yes - both foundation representations beat ResNet-50 on both datasets - but the advantage is statistically significant only on HiRISE; on MSL (600-image test) the CIs include zero.
**6/7. HIRISE - classes helped most by clip vs ResNet-50:** spider (+0.69), slope streak (+0.17), impact ejecta (+0.12). Hardest classes (clip F1):** impact ejecta (0.36), crater (0.66), spider (0.69).
**6/7. MSL - classes helped most by dinov2 vs ResNet-50:** Layers (+0.39), Drill hole (+0.29), Sand (+0.17). Hardest classes (dinov2 F1):** Wheel tracks (0.00), Light-toned veins (0.00), DRT spot (0.17).
**8. Are improvements statistically meaningful?** HiRISE: yes (paired bootstrap p≈0). MSL: no (differences within 95% CI of zero).
**9. Are foundation representations competitive with historical CNN results?** Our best frozen probe reaches HiRISE acc≈0.91 / MSL acc≈0.80, in the neighbourhood of the paper's fine-tuned AlexNet (0.928 / 0.745) - competitive, but see Q10.
**10. Are the historical results actually directly comparable?** No. The paper fine-tuned AlexNet with different preprocessing and (MSL) split; its 96.7%/90.3% include abstention. Our frozen linear probe at 100% coverage is a different protocol; treat Wagstaff numbers as context only.

## Limitations

- Frozen probe, three base-size models; no fine-tuning, no domain FM yet.
- HiRISE originals-only (augmented train images not embedded).
- Near-duplicate (perceptual) leakage beyond exact pixel hashing not exhaustively checked.
- MSL small per-class supports; flagged where tiny.

## Conclusions

Frozen foundation-model features (DINOv2, CLIP) are **competitive with, and on HiRISE significantly better than**, a frozen ImageNet ResNet-50 representation under an identical linear-probe protocol - without any fine-tuning.

- **HIRISE**: best frozen representation = **clip**; foundation vs ResNet-50 advantage is statistically significant.
- **MSL**: best frozen representation = **dinov2**; foundation vs ResNet-50 advantage is NOT statistically significant.

The honest framing: *some modern frozen representations provide competitive frozen features for Martian image classification*; the improvement is dataset-dependent (clear and significant on orbital HiRISE, directional but not significant on the small MSL test set). This is evidence, not a blanket claim that foundation models beat CNNs.
