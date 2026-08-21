# Foundation-Model vs. CNN Features for Mars Image Classification

This project evaluates how well frozen, pretrained image features classify Martian imagery,
comparing a conventional ImageNet CNN, ResNet-50, against two modern foundation models, DINOv2 and
CLIP, on orbital HiRISE and rover MSL data. Each model is used as a fixed feature extractor with a
simple linear classifier trained on top, so the comparison isolates the quality of the representation
rather than any downstream training.

## Motivation

Planetary missions return far more imagery than can be labeled by hand, so representations that
perform well without task-specific training are of practical value; a new HiRISE campaign, for
example, may yield thousands of unlabeled crops alongside only a few labeled examples. Foundation
models such as DINOv2 and CLIP provide general-purpose image features, and the central question is
whether those features outperform a standard ImageNet CNN on Mars imagery in the absence of any
fine-tuning.

The comparison holds everything except the representation fixed. Each model is frozen, each image
passes through that model's official preprocessing, and a single L2-regularized logistic regression is
trained on the resulting features. Two failure modes common to Mars classifiers receive explicit
attention: source-image leakage, since many HiRISE landmark crops originate from the same parent
observation, and pronounced class imbalance. Accordingly, the splits are source-grouped, a leakage
audit runs before any training, and Macro-F1 serves as the primary metric in place of raw accuracy.

## Data

- **HiRISE.** Orbital imagery from the *Mars orbital image (HiRISE) labeled data set, v3.2*,
  [Zenodo 4002935](https://zenodo.org/records/4002935). Eight landmark classes: crater, bright dune,
  dark dune, slope streak, impact ejecta, swiss cheese, spider, and other. The set contains 10,815
  original landmark crops with a released source-grouped train/validation/test split of
  6,997 / 2,025 / 1,793.
- **MSL.** Rover imagery from the *Mars surface image (Curiosity rover) labeled data set, v2.1*,
  [Zenodo 4033453](https://zenodo.org/records/4033453). Nineteen science and engineering classes
  acquired by the Mastcam left and right cameras and MAHLI, using the official chronological,
  sol-based split.

Both datasets are single-band grayscale browse products, replicated to three channels for the RGB
models without artificial colorization. The pipeline downloads the data; it is not stored in the
repository.

## Methods

1. **Environment and download.** Capture the software and hardware environment, then download and
   checksum-verify both datasets from Zenodo.
2. **Auditing.** Report class counts, image dimensions, and per-split distributions, then run a
   leakage audit confirming that no source image, landmark, or exact-pixel duplicate is shared across
   splits. The pipeline halts if any check fails.
3. **Feature extraction.** Pass every image once through each frozen model to produce deterministic
   embeddings, stored with full metadata.
4. **Linear probe.** Fit a StandardScaler on the training set alone, followed by multinomial logistic
   regression, selecting the regularization strength and class weighting on the validation set and
   evaluating the test set only once.
5. **Analysis.** Paired bootstrap significance tests, PCA and UMAP projections, nearest-neighbor
   retrieval, calibration diagnostics, and error analysis, followed by the figures and written report.

## Key Results

Test-set performance under the identical frozen-feature probe. Both foundation models match or exceed
the ResNet-50 baseline on every metric.

**HiRISE, orbital imagery, 8 classes**

| Representation | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|----------------|----------|-------------------|----------|-------------|
| ResNet-50, ImageNet CNN | 0.879 | 0.683 | 0.608 | 0.876 |
| DINOv2 | 0.897 | 0.834 | 0.714 | 0.904 |
| CLIP | 0.915 | 0.828 | 0.737 | 0.919 |

**MSL, rover imagery, 19 classes**

| Representation | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|----------------|----------|-------------------|----------|-------------|
| ResNet-50, ImageNet CNN | 0.743 | 0.600 | 0.558 | 0.738 |
| DINOv2 | 0.795 | 0.628 | 0.598 | 0.782 |
| CLIP | 0.762 | 0.633 | 0.595 | 0.765 |

- On HiRISE, CLIP reaches 91.5% accuracy and 0.919 weighted-F1, with DINOv2 close behind at 89.7%,
  both improving on the 87.9% ResNet-50 baseline.
- HiRISE is dominated by a single majority class, so overall accuracy sits only modestly above the
  0.83 majority-class baseline. Balanced accuracy and Macro-F1 give a fairer view of minority-class
  performance, and there the separation is larger: balanced accuracy rises from 0.68 for ResNet-50 to
  0.83 for both DINOv2 and CLIP, and a paired bootstrap confirms both Macro-F1 gains are statistically
  significant, with 95% confidence intervals excluding zero and p ≈ 0.
- On MSL, DINOv2 leads at 79.5% accuracy, ahead of CLIP at 76.2% and ResNet-50 at 74.3%. All three sit
  far above the 0.31 majority-class baseline, though the differences among them are small and not
  statistically significant on the 600-image test set.
- CLIP and DINOv2 are statistically indistinguishable, and the leader depends on the dataset: CLIP on
  HiRISE and DINOv2 on MSL. Neither foundation model dominates across both.
- The improvement concentrates in the rare, visually distinctive HiRISE classes such as spider and
  impact ejecta, while the smallest classes remain the most difficult for every model.
- These frozen linear-probe results are not directly comparable to fine-tuned, end-to-end CNNs
  reported in the literature.

Full metric tables, including per-class precision, recall, and F1, are in `results/tables/`, figures in
`results/figures/`, and the written report in `results/reports/`.

## Repo Structure

```
.
├── configs/     # dataset, model, preprocessing, and classifier settings
├── src/         # data loaders, frozen extractors, linear probe, evaluation, plotting
├── scripts/     # numbered, rerunnable pipeline, 00 through 41
├── results/     # audits, metric tables, figures, and the written report
├── requirements.txt
└── README.md
```

## Limitations & Future Work

- All results here derive from a frozen linear probe; fine-tuning the backbones would likely alter the
  ranking and is a natural next step.
- Only three backbones are compared, one CNN and two foundation models. Incorporating a remote-sensing
  or Mars-specific pretrained model would broaden the comparison.
- The MSL result is directional only, as the test set is small and several classes contain few
  examples. Additional data or a dedicated label-efficiency study would strengthen it.
- Near-duplicate detection depends on exact-pixel and hash checks, so subtler perceptual duplicates
  may remain undetected.

## Acknowledgments

Data are provided by NASA/JPL and the HiRISE and MSL science teams through the public Zenodo releases
linked above. The experimental design and dataset handling follow Wagstaff et al., *"Mars Image
Content Classification: Three Years of NASA Deployment and Recent Advances"* (AAAI 2021).
