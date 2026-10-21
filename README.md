# BlueBoat — Do Pretrained Vision Encoders See the Same Seafloor Twice?

**Benchmark of pretrained vision encoders, analysis of cross-survey view consistency, and contrastive fine-tuning on BlueBoat Side-Scan Sonar data.**

BlueBoat research project — University of Salento, Computer Engineering, October 2026.

> This README is the repository's central technical and scientific guide. It consolidates documentation previously spread across multiple `.md` files and is maintained alongside the code and saved experimental results.

> **Public source-code release:** The original XTF dataset, generated outputs, embeddings, figures and model checkpoints are not distributed in this repository.
> The experimental results documented below refer to the original research runs. To reproduce them, obtain the required dataset and follow the installation and execution instructions.

## Quick start

Run these commands from the **repository root**. This repository is public; the original XTF dataset must be obtained separately.

**1. Clone and install (macOS/Linux)**

```bash
git clone https://github.com/Letsgosky7/blueboat_encoder_benchmark_public.git
cd blueboat_encoder_benchmark_public

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**2. Place the dataset and configure its location**

The dataset may reside **outside the repository**. Its root directory must contain an `SSS-XTF/` subdirectory with these exact filenames:

```text
/path/to/BlueBoat-dataset/
└── SSS-XTF/
    ├── 2026-08-04-12-59.xtf       # Control
    ├── 2026-08-05-08-00-1.xtf     # A
    ├── 2026-08-06-07-50.xtf       # B
    └── 2026-08-06-09-10.xtf       # C
```

Create **`configs/local.yaml`** in the repository root's `configs/` directory and set `dataset_root` to the **absolute path of the directory containing `SSS-XTF/`**, not to `SSS-XTF/` itself:

```yaml
dataset_root: /path/to/BlueBoat-dataset
```

The file `configs/local.yaml` is intentionally **Git-ignored** and is not provided in the repository. The dataset-to-filename mapping is defined by `SURVEYS` in [`src/data/dataset_builder.py`](src/data/dataset_builder.py). The builder reads `dataset_root` from [`scripts/build_dataset.py`](scripts/build_dataset.py). **Do not commit your private dataset path or raw sonar files.**

**3. Run the pipeline**

```bash
# First-time build: original XTF -> geometry, patches and manifest.
python -m scripts.build_dataset

# Dataset quality checks (after building the patches).
python -m scripts.audit_dataset
python -m scripts.visualize_dataset

# Frozen pretrained encoder benchmarks (run separately as required).
python -m scripts.resnet50_benchmark
python -m scripts.convnext_tiny_benchmark
python -m scripts.vit_b16_benchmark
python -m scripts.clip_vit_b32_benchmark

# Compare previously computed frozen results.
python -m scripts.benchmark_comparison --with-diagnostics
```

If cached patches, embeddings and analysis outputs are already available, **skip costly XTF reconstruction and/or inference**, and use the cached-analysis commands in [Section 9](#9-installation-and-experiment-reproduction). Contrastive ResNet50 training is a **separate optional step** and writes checkpoints; review Section 9 before starting it. The Chapter 2 and Chapter 4 figures can be regenerated with:

```bash
python -m scripts.generate_chapter2_example_figures
python -m scripts.generate_resnet50_chapter4_figures
```

**4. Run the tests**

```bash
python -m pip install pytest    # pytest is a development tool, not in requirements.txt
python -m pytest -q
```

Alternatively, use `python -m unittest discover -s tests -p 'test_*.py' -q` without installing pytest. See [Section 9](#9-installation-and-experiment-reproduction) for all CLI modes, prerequisites, device selection and output locations.

## Contents

1. [Research objective and main findings](#1-research-objective-and-main-findings)
2. [Pipeline architecture and file map](#2-pipeline-architecture-and-file-map)
3. [Acquisitions, geometry, and dataset construction](#3-acquisitions-geometry-and-dataset-construction)
4. [Preprocessing and NoData masks](#4-preprocessing-and-nodata-masks)
5. [Frozen benchmark of four encoders](#5-frozen-benchmark-of-four-encoders)
6. [Evaluation, metrics, and results](#6-evaluation-metrics-and-results)
7. [PCA, clustering, diagnostics, and translations](#7-pca-clustering-diagnostics-and-translations)
8. [Contrastive ResNet50 fine-tuning](#8-contrastive-resnet50-fine-tuning)
9. [Installation and experiment reproduction](#9-installation-and-experiment-reproduction)
10. [Output files, provenance, and maintenance](#10-output-files-provenance-and-maintenance)
11. [Limitations, interpretation, and future work](#11-limitations-interpretation-and-future-work)

## 1. Research objective and main findings

**Research question:** Can vision encoders pretrained on natural images recognize the same nominal seafloor area across two different **Side-Scan Sonar (SSS)** acquisitions?

The same seafloor feature may have different textures, intensities and acoustic shadows as the heading, sensor side, look azimuth, incidence and range change. Geographic registration is also approximate: a pair linked by coordinates is not necessarily pixel-aligned. The project therefore studies **representations/embeddings**, **retrieval** of the positive patch ID within a gallery, and the contribution of unobserved pixels.

The completed stages are:

1. XTF parsing, survey-leg selection, First Bottom estimation, UTM projection, geocoding, patch extraction, and construction of positive and negative pairs;
2. Frozen comparison of **ResNet50, ConvNeXt-Tiny, ViT-B/16 and CLIP ViT-B/32**;
3. SONAR/MASK-ONLY controls, matched negatives, quality and acquisition-condition analyses, PCA, ARI/NMI clustering, and translation tests;
4. Contrastive ResNet50 fine-tuning with **BN-affine** and **last bottleneck** updates under geographically embargoed splits, comparing numerical SONAR with 8-bit PNG.

**Snapshot statistics:**

| Item | Count |
| --- | ---: |
| XTF recordings | 4 (A, B, C and Control) |
| Nominal cross-survey positive pairs | **25** = A–B **4** + A–C **5** + B–C **16** |
| Individual patches in the reference bank | **81** |
| Negative pairs in the frozen benchmark | **172** = 100 within-site + 72 Control |
| Nominal footprint | **50 × 50 m** |
| Original grid | **200 × 200** at **0.25 m/cell** |
| Encoder input | **224 × 224**, three channels |
| Frozen benchmark directional queries | **50** per model/condition |
| Rankable geographic-protocol queries | **48** per modality/stage/condition |

**Key scientific finding:** the highest SONAR Recall@1 in the frozen benchmark is **22%** (ResNet50), and **34/50** queries fail for all four encoders. In the geographically held-out comparison, ResNet50 Recall@1 remains **25%** for the frozen baseline and both fine-tuning variants. **The 22% and 25% values cannot be compared directly:** gallery sizes differ.

The 25 positive pairs are **nominal GPS/geometric correspondences**. This project **does not** independently verify that acoustic structures actually match, and is **not** a certified geographic localization system.

## 2. Pipeline architecture and file map

```text
Four original XTF files (not distributed in this repository)
  └── scripts/build_dataset.py
      ├── src/data/xtf_reader.py          sonar pings, metadata
      ├── src/geometry/survey.py          survey legs, UTM
      ├── src/geometry/seabed.py          First Bottom, geocoding
      └── src/patches/extraction.py       anchors, patches, negative bank
          ├── outputs/manifest.json
          ├── outputs/tables/             CSV and geometry NPZ
          └── outputs/patches/            positive_pairs_*.npz, patch_bank_*.npz
                 │
                 ├── scripts/audit_dataset.py
                 └── scripts/visualize_dataset.py
                 │
                 ▼
          src/encoders/preprocessing.py
                 │
                 ▼
  scripts/{resnet50,convnext_tiny,vit_b16,clip_vit_b32}_benchmark.py
      ├── src/encoders/                  model wrappers, saved embeddings
      ├── src/benchmark/pipeline.py      extraction, scores, metrics
      ├── src/benchmark/reporting.py     margins and reports
      ├── src/analysis/retrieval.py      exact-ID, tie-aware ranking
      └── outputs/embeddings/            model-specific NPZ and CSV
                 │
                 ├── scripts/benchmark_comparison.py
                 ├── scripts/acquisition_condition_analysis.py
                 ├── scripts/positive_pair_diagnostics.py
                 ├── scripts/embedding_pca.py
                 ├── scripts/cluster_agreement.py
                 └── scripts/registration_sensitivity.py
                 │
                 ▼
  scripts/resnet50_sonar_vs_png.py
      └── src/experiments/resnet50_sonar_png/
          ├── data.py                    SONAR/PNG pairs and inputs
          ├── splits.py                  geographic CV and embargo
          ├── augmentations.py           online augmentations
          ├── training.py                backbone, MLP and InfoNCE
          ├── evaluation.py              held-out ranking
          ├── diagnostics.py             parameter deltas and embeddings
          ├── ablation.py                last-bottleneck follow-up
          └── pipeline.py                orchestration and protocols
                 │
                 ▼
  outputs/analysis/resnet50_cross_survey_cv/
      ├── heldout_summary.csv, training_history.csv, protocol.json
      ├── fold_01/, fold_02/, fold_03/
      └── new_results/{diagnostics,last_bottleneck}/
                 │
                 └── scripts/generate_resnet50_chapter4_figures.py
```

Reusable code lives in [`src/`](src), executable entry points in [`scripts/`](scripts), and tests in [`tests/`](tests). Numbered scripts under [`scripts/legacy/`](scripts/legacy) are **historical/exploratory** and are not required by the canonical pipeline.

**Key module index:**

| File | Responsibility |
| --- | --- |
| [`src/data/xtf_reader.py`](src/data/xtf_reader.py) | XTF decoding with `pyxtf` and navigation metadata |
| [`src/data/dataset_builder.py`](src/data/dataset_builder.py) | Dataset construction sequence and manifest |
| [`src/geometry/survey.py`](src/geometry/survey.py) | Straight survey legs and UTM positions |
| [`src/geometry/seabed.py`](src/geometry/seabed.py) | First Bottom estimation, temporal tracking, ground range |
| [`src/patches/extraction.py`](src/patches/extraction.py) | Anchors, mutual-nearest matching, rasterization, negatives |
| [`src/encoders/preprocessing.py`](src/encoders/preprocessing.py) | Robust scaling and mask-aware resizing |
| [`src/encoders/embedding_store.py`](src/encoders/embedding_store.py) | Embedding serialization and provenance checks |
| [`src/benchmark/pipeline.py`](src/benchmark/pipeline.py) | Shared four-encoder benchmark pipeline |
| [`src/benchmark/reporting.py`](src/benchmark/reporting.py) | Score statistics and matched-negative margins |
| [`src/analysis/retrieval.py`](src/analysis/retrieval.py) | Retrieval, ranking, ties, and Recall@k |
| [`src/analysis/dataset_quality.py`](src/analysis/dataset_quality.py) | Mask and NoData audits |
| [`src/analysis/embedding_pca.py`](src/analysis/embedding_pca.py) | PCA on frozen features |
| [`src/analysis/cluster_agreement.py`](src/analysis/cluster_agreement.py) | ARI/NMI for Ward clusters |
| [`src/analysis/registration_sensitivity.py`](src/analysis/registration_sensitivity.py) | Controlled 0–20 m shifts |
| [`src/experiments/resnet50_sonar_png/`](src/experiments/resnet50_sonar_png) | Full contrastive fine-tuning workflow |

## 3. Acquisitions, geometry, and dataset construction

### 3.1 Source data

BlueBoat is an unmanned surface vehicle (USV) equipped with a **Cerulean Omniscan 450 Side-Scan Sonar**. Its XTF recordings contain intensities from the **port** (left) and **starboard** (right) channels, alongside metadata such as `SensorXcoordinate`, `SensorYcoordinate`, `SensorHeading` and timestamps. The [XTF reader](src/data/xtf_reader.py) arranges the intensities into ping × range arrays. This pipeline does not decode `.svlog/.svlz` files, which have not been established as necessary for the selected XTF acquisitions.

The filename mapping is defined by the `SURVEYS` constant in [`src/data/dataset_builder.py`](src/data/dataset_builder.py):

| Survey | XTF filename |
| --- | --- |
| Control | `2026-08-04-12-59.xtf` |
| A | `2026-08-05-08-00-1.xtf` |
| B | `2026-08-06-07-50.xtf` |
| C | `2026-08-06-09-10.xtf` |

**The four raw XTF files are not distributed with the repository.** They must be provided separately to reconstruct the dataset. Control is used for negative comparisons; it is not a fourth positive repeat pass. Place the files under `<dataset_root>/SSS-XTF/` and specify `dataset_root` in the local `configs/local.yaml` file, as shown in [Quick start](#quick-start).

Source coordinates are WGS84 longitude/latitude (EPSG:4326). [`project_tracks_to_utm`](src/geometry/survey.py) projects them to metric UTM coordinates (**EPSG:32633**, UTM zone 33N, for these acquisitions). The code selects the EPSG code from the geographic position and checks consistency across surveys.

### 3.2 Identifying usable survey legs

[`detect_survey_legs`](src/geometry/survey.py) samples the navigation every **20 rows**, estimates course (direction of travel), speed and turning, and identifies sufficiently regular segments:

| Parameter | Value |
| --- | ---: |
| Maximum local turn | **8°** |
| Minimum speed | **0.25 m/s** |
| Short gap allowed for merging | **5 s** |
| Minimum segment duration | **20 s** |
| Minimum net displacement | **25 m** |
| Minimum straightness, endpoint distance / path length | **0.90** |

This selection excludes unsuitable turns and reversals. Sensor **heading** and vehicle **course** are distinct quantities. The selected survey legs are saved in `outputs/tables/survey_legs.csv`.

### 3.3 First Bottom: from the water column to the seafloor

**First Bottom** approximates the first return attributable to the seafloor and separates the water column. [`src/geometry/seabed.py`](src/geometry/seabed.py) processes port and starboard independently:

1. Blocks of **200 pings**, advanced by **100** pings;
2. Median profile, `log1p(x)=ln(1+x)` transform and Gaussian smoothing with **σ=3**;
3. Search for transitions at **0.4–10 m** slant range, requiring greater mean intensity after each candidate;
4. Temporal selection via **dynamic programming**, trading transition strength against smoothness across blocks, with **λ=0.05** and **d0=0.50 m**;
5. Exclusion of estimates scoring below **0.30** from high-confidence geometry.

The transition score measures how well a candidate explains the profile change; it is **not** a statistical probability that the First Bottom is correct.

For slant range `r` and estimated height `h` above a locally flat bottom:

```text
g = sqrt(r² - h²),   for r ≥ h
```

From sensor UTM position `(Es, Ns)`, ground range `g`, heading `θ` and side:

```text
E = Es ± g cos(θ)
N = Ns ∓ g sin(θ)
```

The upper signs correspond to starboard and the lower signs to port. **These are estimated coordinates**, dependent on orientation, navigation and the seafloor model, and do not guarantee actual alignment between surveys.

### 3.4 Georeferenced anchors and positive pairs

[`src/patches/extraction.py`](src/patches/extraction.py) samples anchors every **20 pings** and at **1 m** ground-range intervals. Anchors retain metric position, side, heading, look azimuth, incidence and confidence. Two surveys are matched using **mutual nearest neighbours**: an anchor in A and one in B must be each other's nearest counterpart in the reconstructed UTM system.

With an initial **10 m** threshold, the geometric candidate counts are **5,880** for A–B, **4,123** for A–C and **10,437** for B–C. These are **not** the final positive images. Subsequent selection requires high confidence, nominal separation no greater than **2 m**, and at least **25 m** between selected centers.

Two mosaics are built around each center on the same nominal geographic grid: **50 × 50 m**, **0.25 m per cell**, or **200 × 200** samples. Mean `log1p` intensity in each cell is computed from valid samples, while a sample count and Boolean validity mask are stored. A mosaic may combine **multiple pings and both SONAR sides**; the selected anchor's geometry metadata does not necessarily represent every pixel.

Accepting a pair requires **at least 60% valid coverage in each patch** and **at least 50% joint coverage**:

| Survey pair | Nominal positives |
| --- | ---: |
| A–B | 4 |
| A–C | 5 |
| B–C | 16 |
| **Total** | **25** |

**Nominal** means that the association is obtained from acquisition georeferencing, without independent acoustic registration. The 2 m anchor threshold **does not quantify the maximum GPS error**.

### 3.5 Negative bank (frozen benchmark only)

A separate bank contains **81 observations** from A, B, C and Control. The frozen benchmark forms **100 non-colocated within-site negatives** (four per positive, with centers at least **75 m** apart and approximately comparable geometry), plus **72 negatives** involving Control (24 for each A/B/C pass): **172 in total**.

These pairs support cosine distributions and matched-negative margins. They **are not part of the retrieval galleries** and **are not reused as the fine-tuning training dataset**. Contrastive negatives are selected dynamically among geographically separated pairs in the *training fold*.

Derived source artifacts include `outputs/manifest.json`, `outputs/tables/positive_pair_metadata.csv`, `outputs/tables/null_pair_metadata.csv`, `outputs/patches/positive_pairs_*.npz` and `outputs/patches/patch_bank_*.npz`.

#### Run this stage — XTF to geocoded patches

**Prerequisites:** the four original XTF files and `configs/local.yaml` configured as described above. Run from the repository root:

```bash
python -m scripts.build_dataset
python -m scripts.audit_dataset
python -m scripts.visualize_dataset
python -m scripts.generate_chapter2_example_figures
```

**Outputs:** `outputs/manifest.json`, `outputs/tables/`, `outputs/patches/` and diagnostic figures in `outputs/figures/`. The last three commands operate on the saved patches; **skip `build_dataset` if the required patch artifacts are already available**.

## 4. Preprocessing and NoData masks

[`prepare_sonar_rgb`](src/encoders/preprocessing.py) independently transforms each patch, without using its paired counterpart:

1. Robust normalization using the **1st and 99.5th percentiles of valid pixels only**;
2. Intensity scaling and clipping to **[0, 1]**;
3. **200×200 → 224×224** resizing with **mask-aware** interpolation to prevent missing pixels from diluting SONAR intensities;
4. Filling of NoData cells with the median normalized intensity while retaining a separate Boolean mask;
5. Replication of the single channel into **three channels**, followed by checkpoint-specific normalization (ImageNet or CLIP) in each encoder.

Benchmark conditions:

- **SONAR:** actual acoustic intensities, with NoData pixels filled;
- **MASK-ONLY:** only each patch's *individual* validity mask (valid=1, NoData=0), without SONAR intensities.

MASK-ONLY is a **confounder control**. High scores without acoustic texture suggest that observed-region geometry may convey usable cues, but do **not** establish that an encoder relies exclusively on masks for every match.

For fine-tuning, [`data.py`](src/experiments/resnet50_sonar_png/data.py) also processes two representations of **the same normalized intensities**:

- **Numerical SONAR:** float32 values;
- **8-bit PNG:** grayscale PNG encoding with **256 levels**, followed by decoding. This is neither another sensor source nor a photographic image; the main difference is quantization.

#### How preprocessing runs in practice

The preprocessing function in [`src/encoders/preprocessing.py`](src/encoders/preprocessing.py) is **a shared library function, not a standalone CLI**. It is called as part of the four benchmark entry points below. You do not need to preprocess and save the 224×224 images separately.

## 5. Frozen benchmark of four encoders

**Frozen** means inference only, with **no updates to pretrained weights**. All four encoder implementations reside in [`src/encoders/`](src/encoders), with separate CLIs and a shared [benchmark pipeline](src/benchmark/pipeline.py).

| Encoder | Weights / architecture | Extracted features | Dimension |
| --- | --- | --- | ---: |
| [ResNet50](src/encoders/resnet50.py) | torchvision `IMAGENET1K_V2` | Global Average Pooling, final `fc` replaced with `Identity()` | **2048** |
| [ConvNeXt-Tiny](src/encoders/convnext_tiny.py) | ImageNet-1K V1 | Backbone output, without final classifier | **768** |
| [ViT-B/16](src/encoders/vit_b16.py) | ImageNet-1K V1 | `CLS` token representation | **768** |
| [CLIP ViT-B/32](src/encoders/clip_vit_b32.py) | OpenAI `ViT-B-32-quickgelu` | Visual encoder only; no text or prompts | **512** |

For CLIP, preserving **QuickGELU** is essential. [`embedding_store.py`](src/encoders/embedding_store.py) records and validates embedding dimensions and provenance, and rejects variants incompatible with archived results.

Patch embeddings are L2-normalized before dot products are computed. **Absolute cosine scores from different models are not calibrated to the same semantic scale**: a CLIP score of 0.95 is not necessarily better for retrieval than a ResNet50 score of 0.75.

CLI entry points: [`scripts/resnet50_benchmark.py`](scripts/resnet50_benchmark.py), [`scripts/convnext_tiny_benchmark.py`](scripts/convnext_tiny_benchmark.py), [`scripts/vit_b16_benchmark.py`](scripts/vit_b16_benchmark.py), and [`scripts/clip_vit_b32_benchmark.py`](scripts/clip_vit_b32_benchmark.py).

#### Run this stage — frozen embedding extraction

**Prerequisites:** `outputs/patches/` and associated dataset metadata from Stage 3. The first invocation may download pretrained weights. Run one or all encoders:

```bash
python -m scripts.resnet50_benchmark
python -m scripts.convnext_tiny_benchmark
python -m scripts.vit_b16_benchmark
python -m scripts.clip_vit_b32_benchmark
```

**Outputs:** model-specific embedding NPZs, scores, margins and retrieval CSVs in `outputs/embeddings/`. For previously generated data, the `--retrieval-only` and `--analyze-only` modes can avoid repeating inference; see [Section 9.3](#93-running-the-frozen-encoders).

## 6. Evaluation, metrics, and results

### 6.1 Cosine similarity and margins

For embeddings `u, v`, cosine similarity is `(u·v)/(||u||₂||v||₂)`; after L2 normalization, it equals their dot product. It is **not** a probability that the physical locations match.

A pair's **matched-negative margin** is its positive cosine minus the **mean** cosine of its four geometry-matched negative examples. A margin greater than zero means the positive is more similar *on average* than the selected negatives; it **does not guarantee** successful gallery ranking. Implementation: [`src/benchmark/reporting.py`](src/benchmark/reporting.py).

### 6.2 Bidirectional closed-set retrieval and ties

[Retrieval ranking](src/analysis/retrieval.py) queries a patch from survey A against candidates from B, then repeats the reverse direction B→A. Galleries contain **only the selected positives**, not the 81-patch reference bank. A query succeeds if the nominal counterpart's ID appears among the first `k` candidates.

| Group | Gallery size | Directional queries | Chance Recall@1 |
| --- | ---: | ---: | ---: |
| A–B | 4 | 8 | 25% |
| A–C | 5 | 10 | 20% |
| B–C | 16 | 32 | 6.25% |
| **Overall** | Varies | **50** | **12%** (weighted) |

The expected **Recall@3 chance baseline is 36%**. The benchmark evaluates matching the **dataset-defined ID**, not physical GPS localization accuracy.

Ranking is **tie-aware**: when vectors or scores are equal within tolerance, fractional credit is distributed across tied ranks, avoiding artificially high Recall from arbitrary ordering.

**50 directional queries are not 50 independent regions**: they represent 25 pairs evaluated in both directions.

### 6.3 Frozen benchmark results

Source: `outputs/analysis/frozen_encoders/four_encoder_metrics_summary.csv`; aggregation in [`scripts/benchmark_comparison.py`](scripts/benchmark_comparison.py).

| Encoder | SONAR R@1 | SONAR R@3 | MASK-ONLY R@1 | MASK-ONLY R@3 |
| --- | ---: | ---: | ---: | ---: |
| **ResNet50** | **22%** | 42% | 28% | 54% |
| ConvNeXt-Tiny | 14% | 40% | 14% | 48% |
| ViT-B/16 | 14% | 42% | 22% | 38% |
| CLIP ViT-B/32 | 8% | 30% | 16% | 44% |
| **Chance** | **12%** | 36% | 12% | 36% |

ResNet50 achieves the highest SONAR Recall@1, although performance remains limited. **34/50** SONAR queries are incorrectly ranked first by **all four encoders**, and none is correctly retrieved by all four. Shared failures are consistent with coverage artifacts, domain difficulty and uncertain nominal matches, but the evidence does not isolate a single cause.

The **MASK-ONLY** outcome is a methodological warning: ResNet50 achieves **28% R@1 without SONAR intensities**, versus **22% with SONAR**. In this dataset, validity-mask geometry may influence some scores. Do not conclude automatically that the networks are genuinely recognizing the seafloor.

#### Run this stage — cross-encoder evaluation

**Prerequisites:** cached scores/retrieval results for **all four** frozen encoders (Stage 5). To generate comparative tables and plots without running an encoder:

```bash
python -m scripts.benchmark_comparison
```

For additional shared-error diagnostics, also provide patch NPZs and quality/negative metadata from Stage 3:

```bash
python -m scripts.benchmark_comparison --with-diagnostics
```

**Outputs:** `outputs/analysis/frozen_encoders/`, including `four_encoder_metrics_summary.csv` and comparison figures. The optional diagnostics require additional cached inputs but **no training**.

## 7. PCA, clustering, diagnostics, and translations

### 7.1 Pair and quality audits

[`scripts/audit_dataset.py`](scripts/audit_dataset.py), [`src/analysis/dataset_quality.py`](src/analysis/dataset_quality.py), and [`scripts/positive_pair_diagnostics.py`](scripts/positive_pair_diagnostics.py) combine coverage, connected NoData regions, geometry, and encoder results.

The `requires_review` flag is raised when central joint coverage is below **80%**, or a connected NoData region occupies more than **25%** of either patch. **9 of the 25 pairs** require review by these criteria; they are not automatically incorrect.

In an **exploratory analysis conducted after seeing the results**, the **10 pairs** with joint coverage below **60%** had **zero** aggregate Recall@1 successes across the four encoders. The 60% value is **not a validated physical threshold**. Outputs include `outputs/tables/positive_pair_quality.csv` and the `outputs/analysis/positive_pair_diagnostics` directory.

**Run the pair-quality audit and retrieval diagnostics:**

```bash
# First requires the positive patch NPZs; writes positive_pair_quality.csv.
python -m scripts.audit_dataset

# Requires the quality CSV, positive metadata and frozen comparison diagnostics.
python -m scripts.positive_pair_diagnostics
```

The second command writes into `outputs/analysis/positive_pair_diagnostics/`. Run the frozen comparison with `--with-diagnostics` first if its diagnostics are not already present.

### 7.2 Acquisition conditions

[`scripts/acquisition_condition_analysis.py`](scripts/acquisition_condition_analysis.py) studies margins and retrieval relative to anchor **heading** (recorded orientation), **look azimuth** (horizontal viewing direction), **incidence angle** under the simplified model, **ground range**, and **side**.

For **20/25** pairs, the anchors' look-azimuth difference is at least **160°**. Their retrieval outcomes are worse in the analyzed dataset, but the groups also differ by survey and gallery size; this **does not establish a causal effect of angle**. Furthermore, an anchor does not characterize every pixel, because a mosaic may combine port and starboard samples.

Results and figures reside in `outputs/analysis/acquisition_conditions`, including `acquisition_heading_look_vs_margin.png` and `acquisition_sonar_mask_delta_vs_geometry.png`.

**Run acquisition-condition analysis** once the positive metadata and four encoder result sets have been computed:

```bash
python -m scripts.acquisition_condition_analysis
```

This reads cached metadata and margins/retrieval CSVs, and writes diagnostic CSV/PNG files under `outputs/analysis/acquisition_conditions/`. It does **not** rerun encoders.

### 7.3 Embedding PCA

[`src/analysis/embedding_pca.py`](src/analysis/embedding_pca.py), invoked by [`scripts/embedding_pca.py`](scripts/embedding_pca.py), runs PCA on the **50 observations** (two patches for each of the 25 pairs), separately for each **model × condition**. It examines the first two principal components, linked patch IDs in scatter plots and distances in projected space. **Retrieval still uses the original embeddings**, not the projected coordinates.

| Encoder | Variance explained by 2 PCs, SONAR | Variance explained by 2 PCs, MASK-ONLY |
| --- | ---: | ---: |
| ResNet50 | 35.99% | 44.19% |
| ConvNeXt-Tiny | 46.24% | 44.55% |
| ViT-B/16 | 41.59% | 39.83% |
| CLIP ViT-B/32 | 34.15% | 50.12% |

PCA axes from different models are not directly comparable. **Explained variance ≠ correct-patch retrieval capability.** Outputs are in `outputs/analysis/final_evaluation/embeddings/pca`, notably `pca_variance.csv`, `pca_coordinates.csv`, and per-encoder figures.

**Run PCA** after the frozen embedding NPZs and positive metadata/quality CSVs are available:

```bash
python -m scripts.embedding_pca
# Optional subset:
python -m scripts.embedding_pca --models resnet50 vit_b16
```

Results are written under `outputs/analysis/final_evaluation/embeddings/pca/` without recomputing embeddings or changing retrieval results.

### 7.4 Ward clustering and cross-survey stability

[`src/analysis/cluster_agreement.py`](src/analysis/cluster_agreement.py) and [`scripts/cluster_agreement.py`](scripts/cluster_agreement.py) apply hierarchical **Ward clustering** to the **original L2-normalized embeddings**, separately for each survey in an A–B/A–C/B–C combination. The resulting partitions of the same nominal IDs are compared using **ARI** (*Adjusted Rand Index*, corrected for chance agreement and potentially negative) and **NMI** (*Normalized Mutual Information*, in [0,1]).

Configurations: A–B (4 pairs) uses `k=2,3`; A–C (5 pairs) and B–C (16 pairs) use `k=2,3,4` where admissible. **64 configurations** were valid in the reference run. High ARI/NMI means similar grouping structure between passes, **not** correct individual matches.

Outputs reside in `outputs/analysis/final_evaluation/embeddings/clustering`: `cluster_agreement.csv`, assignments, provenance, and heatmaps.

**Run clustering** on the saved original frozen embeddings (not on PCA coordinates):

```bash
python -m scripts.cluster_agreement
# Optional subset and cluster limit:
python -m scripts.cluster_agreement --models resnet50 vit_b16 --max-clusters 3
```

Outputs are stored under `outputs/analysis/final_evaluation/embeddings/clustering/`; no encoder inference or training is performed.

### 7.5 Registration sensitivity: controlled translations

[`src/analysis/registration_sensitivity.py`](src/analysis/registration_sensitivity.py), invoked by [`scripts/registration_sensitivity.py`](scripts/registration_sensitivity.py), keeps the first survey patch fixed and shifts the **second patch eastward** by `0, 2, 5, 10, 20 m`, corresponding to `0, 8, 20, 40, 80` columns at 0.25 m/cell.

This is a **finite-crop** experiment: content beyond the boundary is lost, new cells become NoData, and there is no wraparound. At 0 m it reuses archived frozen embeddings; shifts greater than zero require **new inference**, but **no training**. Gallery sizes remain 4/5/16.

| Encoder | SONAR R@1 at 0 m | SONAR R@1 at 20 m |
| --- | ---: | ---: |
| ResNet50 | 22% | 16% |
| ConvNeXt-Tiny | 14% | 10% |
| ViT-B/16 | 14% | 12% |
| CLIP ViT-B/32 | 8% | 12% |

Mean valid coverage in the second patch drops from **78.91% to 44.29%**, and joint coverage from **64.68% to 31.93%**. The experiment changes *both position and available content*; it **does not isolate a field-measured georeferencing error**. Outputs: `outputs/analysis/final_evaluation/registration`, `registration_summary.csv`, `registration_queries.csv` and `registration_coverage.csv`.

**Run the controlled-translation test** using the saved patches and frozen embeddings:

```bash
# Small check: ResNet50, baseline and a single nonzero shift.
python -m scripts.registration_sensitivity --models resnet50 --shifts 0 2

# Complete sensitivity analysis:
python -m scripts.registration_sensitivity
```

The zero-shift baseline uses cached vectors, but **nonzero shifts rerun frozen inference**, requiring model weights. No training occurs. Results are under `outputs/analysis/final_evaluation/registration/`.

## 8. Contrastive ResNet50 fine-tuning

Main implementation: [`src/experiments/resnet50_sonar_png/`](src/experiments/resnet50_sonar_png), entry point: [`scripts/resnet50_sonar_vs_png.py`](scripts/resnet50_sonar_vs_png.py). This is **not purely self-supervised**: positive pairs are supplied by georeferencing. Fine-tuning is therefore **weakly supervised contrastive learning**, using nominal identities that have not been acoustically verified.

#### Run this stage — inspect folds before training

**Prerequisites:** positive pair NPZs and their metadata built in Stage 3. Inspect the geographic split and embargo **without loading a model or training**:

```bash
python -m scripts.resnet50_sonar_vs_png --audit-only
```

Only proceed with training if the split is valid and suitable for the intended experiment.

### 8.1 Dataset formation: 25 pairs, no new independent correspondences

[`data.py`](src/experiments/resnet50_sonar_png/data.py) reads the same three `positive_pairs_*.npz` archives and validates counts **4/5/16**, IDs, **200×200** dimensions, the nominal **50 m** footprint, individual coverage ≥**60%**, and joint coverage ≥**50%**. It adds no new pairs, external data, isolated-bank patches, or any of the original 172 frozen-benchmark negatives.

The **numerical SONAR** (float32) and **8-bit grayscale PNG** modalities encode/decode the same normalized intensities, so they test quantization rather than differences between sensors.

[`augmentations.py`](src/experiments/resnet50_sonar_png/augmentations.py) applies **online**, while constructing batches, independent pseudorandom perturbations to each view of a pair:

| Transformation | Configuration |
| --- | --- |
| Contrast and intensity factor | Multipliers in **[0.85, 1.15]** |
| Brightness offset | **[−0.08, +0.08]** |
| Gaussian noise | Standard deviation **0.025** |
| Translation | **±3 pixels per axis** |
| Arbitrary rotations/flips | **Not used** |
| NoData during augmentation | Sampling from valid intensities of the same observation |

Augmentations increase view diversity, **not** the number of independent geographic regions. PNGs stored in `fold_*/png/train_png/` are quantized copies of training observations, not necessarily every online-augmented image from every epoch. Variable NoData filling can weaken shortcuts caused by constant fills, but does not prove that all shortcuts disappear.

### 8.2 ResNet50 backbone, MLP, and InfoNCE

ResNet50 `IMAGENET1K_V2` replaces its final classifier with `fc=Identity()`, exposing **2048-dimensional Global Average Pooling** features, `h = fθ(x)`. During training, a two-layer **projection head** with ReLU, `2048→256→128`, yields `z̃=gφ(h)` followed by `z=z̃/||z̃||₂`. The projection head has **557,440 trainable parameters**.

[`training.py`](src/experiments/resnet50_sonar_png/training.py) uses **symmetric InfoNCE with temperature τ=0.2** on the normalized **128-dimensional** `z` vectors. For each anchor `i`, its positive `p(i)` comes from the other acquisition of the same pair; its negatives are other batch observations that are **sufficiently far apart geographically**. The anchor itself and nearby pairs are excluded from the denominator. Both pair directions serve as anchors.

```text
L = -(1/|A|) Σ_(i in A) log[
      exp(z_i · z_p(i) / τ) /
      Σ_(j in V_i) exp(z_i · z_j / τ)
    ],   τ = 0.2
```

`A` contains anchors with at least one valid negative. `V_i` includes the positive and only eligible negatives. Batches contain up to **4 cross-survey pairs** (8 images) and at least two geographically separated pairs. The loss encourages the positive to be relatively more similar than negatives; it **does not verify that GPS-derived labels are physically correct**.

**At evaluation time, the projection head is discarded:** only the backbone's **L2-normalized 2048-D features** are used. Augmentations are not applied during evaluation.

### 8.3 Purged geographic cross-validation

[`splits.py`](src/experiments/resnet50_sonar_png/splits.py) orders positions along the main UTM axis and forms **three contiguous blocks**. In each fold, one block is held out, while training excludes other pairs near its nominal footprint: a **25 m embargo** beyond the nominal **50×50 m** patch squares. This prevents nominally overlapping locations from appearing in train and test within the same fold, but does not establish exact acoustic alignment.

| Fold | Training | Embargoed | Test | Rankable directional queries |
| --- | ---: | ---: | ---: | ---: |
| 1 | 13 | 3 | 9 | 18 |
| 2 | 6 | 11 | 8 | 14 |
| 3 | 12 | 5 | 8 | 16 |
| **Total held out** | – | – | **25** | **48** |

Every pair is tested once. Held-out groups with only one candidate **do not contribute to Recall@k**, where success would be trivial, but remain included in mean positive cosine. Galleries are defined **by fold and survey pair**, without mixing representations generated by models trained on different folds. Fold 2 has particularly few valid negatives.

The geographic-protocol chance baseline is **16.67% R@1** and **50% R@3**, **not directly comparable** to the 12%/36% chance levels or the 22% ResNet50 result from the full 50-query benchmark.

### 8.4 BN-affine and last bottleneck

| Configuration | Optimized ResNet50 parameters | Projection-MLP parameters | BN running mean/variance |
| --- | ---: | ---: | --- |
| **BN-affine** | **6,144**: scales/biases in the three BatchNorm layers of `layer4[-1]` | **557,440** | Frozen |
| **Last bottleneck** | **4,462,592**: entire `layer4[-1]`, including convolutions | **557,440** | Frozen |

Earlier blocks remain frozen. `model.eval()` keeps running BatchNorm statistics fixed while allowing gradients through unlocked parameters. Both experiments use **AdamW**, `lr=1e-5`, `weight_decay=1e-4`, **3 epochs**, at most **4 pairs per batch**, and **seed 42** incremented by fold. SONAR and PNG modalities each start from the same pretrained ImageNet weights.

The **last bottleneck** variant was selected **after inspecting the first held-out results**, making it an **exploratory ablation**, not an independent statistical confirmation. Implementations: [`training.py`](src/experiments/resnet50_sonar_png/training.py), [`ablation.py`](src/experiments/resnet50_sonar_png/ablation.py), [`pipeline.py`](src/experiments/resnet50_sonar_png/pipeline.py), [`evaluation.py`](src/experiments/resnet50_sonar_png/evaluation.py).

### 8.5 Held-out results and limitations

Sources: `outputs/analysis/resnet50_cross_survey_cv/heldout_summary.csv` for frozen and BN-affine, and `outputs/analysis/resnet50_cross_survey_cv/new_results/last_bottleneck/heldout_summary.csv` for the follow-up.

| Modality | Stage | R@1 | R@3 | Mean positive cosine |
| --- | --- | ---: | ---: | ---: |
| Numerical SONAR | Frozen | **25.00%** | 54.17% | 0.709724 |
| Numerical SONAR | BN-affine | **25.00%** | 54.17% | 0.709729 |
| Numerical SONAR | Last bottleneck | **25.00%** | 52.08% | 0.710531 |
| 8-bit PNG | Frozen | **25.00%** | 52.08% | 0.709826 |
| 8-bit PNG | BN-affine | **25.00%** | 52.08% | 0.709831 |
| 8-bit PNG | Last bottleneck | **25.00%** | 52.08% | 0.710610 |

In **every variant**, Recall@1 remains **25%**: small cosine changes do not produce additional rank-1 successes. The ablation changes only **4 ranks across 192 paired comparisons**, with no new top-1 successes; those comparisons are correlated rather than independent geographic regions. In the **MASK-ONLY** control, R@1 is **29.17%** and R@3 is **62.50%** for all variants, consistent with possible coverage-related shortcuts.

Three-epoch loss curves are not monotonic, and **fold 2** produces only **one useful batch per epoch**. On these data, no retrieval improvement is observed following fine-tuning. The positive pairs are few, geographically correlated and nominal; effects due to NoData and geometry remain difficult to disentangle. These results **do not establish a general inability** of ResNet50 or all pretrained models to process sonar imagery.

Diagnostics are stored in `outputs/analysis/resnet50_cross_survey_cv/new_results/diagnostics`, including parameter changes, embedding drift, positive-cosine shifts and negative margins. The follow-up figures can be generated from saved CSVs without retraining.

#### Run fine-tuning, diagnostics and report figures

**Training is optional and may write new checkpoints/results.** After the successful split audit, on an Apple Silicon Mac:

```bash
# BN-affine training: 3 geographically held-out folds.
python -m scripts.resnet50_sonar_vs_png \
  --folds 3 --buffer-m 25 --epochs 3 \
  --batch-size 4 --lr 1e-5 --seed 42 --device mps

# Requires saved BN-affine checkpoints; no retraining:
python -m scripts.resnet50_sonar_vs_png --diagnose-only --device mps

# Optional exploratory last-bottleneck follow-up:
python -m scripts.resnet50_sonar_vs_png \
  --ablation-last-bottleneck --folds 3 --epochs 3 \
  --batch-size 4 --lr 1e-5 --seed 42 --device mps

# Only after both sets of training CSVs exist; no training:
python -m scripts.generate_resnet50_chapter4_figures
```

The regular training results are in `outputs/analysis/resnet50_cross_survey_cv/`; the ablation is isolated under `new_results/last_bottleneck/`, and its plots under `chapter4_figures/`. Use `--device cpu` or `--device cuda` where appropriate. For cautions about checkpoint overwrites and full CLI modes, see [Section 9.5](#95-contrastive-resnet50-audit-training-diagnostics-ablation).

## 9. Installation and experiment reproduction

### 9.1 Prerequisites and virtual environment

This public repository distributes source code only. Original XTF files, generated outputs, embeddings and model checkpoints are not included.

[`requirements.txt`](requirements.txt) includes NumPy, pandas, Matplotlib, pyxtf, PyYAML, pyproj, SciPy, torch, torchvision, open_clip_torch and Pillow. Versions are **not pinned**: for exact reproducibility, record Python/package versions, operating system and inference/training device. The first run may download pretrained model weights.

```bash
git clone https://github.com/Letsgosky7/blueboat_encoder_benchmark_public.git
cd blueboat_encoder_benchmark_public

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Built-in test runner (no extra test dependency).
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run commands from the **project root**, not from `scripts/`. Do not commit credentials or machine-specific paths. On Windows, the virtual-environment activation command differs (e.g. `.venv\Scripts\activate` in Command Prompt).

### 9.2 From the original XTF files to patches (when required)

The external dataset root must contain the following structure:

```text
/path/to/BlueBoat-dataset/
  SSS-XTF/
    2026-08-04-12-59.xtf
    2026-08-05-08-00-1.xtf
    2026-08-06-07-50.xtf
    2026-08-06-09-10.xtf
```

Create **`configs/local.yaml` locally** (it is excluded from Git by [`.gitignore`](.gitignore)):

```yaml
dataset_root: /absolute/path/to/BlueBoat-dataset
```

**Important:** `dataset_root` names the parent of `SSS-XTF/`, not the `SSS-XTF/` directory. The dataset can be anywhere on disk; placing it inside the repository is unnecessary. The exact four input filenames are specified by `SURVEYS` in [`src/data/dataset_builder.py`](src/data/dataset_builder.py). [`scripts/build_dataset.py`](scripts/build_dataset.py) loads this YAML file and passes the path to `DatasetBuilder`.

Then run:

```bash
python -m scripts.build_dataset
python -m scripts.audit_dataset
python -m scripts.visualize_dataset

# Rebuild reproducible Chapter 2 examples from the saved patch NPZs.
python -m scripts.generate_chapter2_example_figures
```

The [builder](src/data/dataset_builder.py) produces `outputs/tables/`, `outputs/patches/` and `outputs/manifest.json`. XTF processing, geometry and patch construction can be expensive. **If the saved original outputs are already available, rebuilding them is not necessary just to generate analysis reports or figures.** The figure generator still needs its saved patch inputs.

### 9.3 Running the frozen encoders

```bash
python -m scripts.resnet50_benchmark
python -m scripts.convnext_tiny_benchmark
python -m scripts.vit_b16_benchmark
python -m scripts.clip_vit_b32_benchmark
```

Each benchmark supports the following modes, provided the required cached files are present:

| CLI mode | Purpose |
| --- | --- |
| No flags | Frozen embedding extraction, scores, retrieval and reports |
| `--analyze-only` | Reprocess previously computed CSV results |
| `--verify-embeddings` | Check cosine-score consistency against saved NPZ embeddings |
| `--retrieval-only` | Compute rankings from saved embeddings, **without inference** |

Outputs include `outputs/embeddings/{model}_embeddings.npz`, `{model}_sonar_mask_pair_scores.csv`, `{model}_matched_null_margins.csv` and `{model}_retrieval_{summary,queries,candidates}.csv`.

### 9.4 Analysis of cached results

```bash
# Compare all four encoders; export comparative figures and tables.
python -m scripts.benchmark_comparison --with-diagnostics

# Acquisition geometry and quality diagnostics.
python -m scripts.acquisition_condition_analysis
python -m scripts.positive_pair_diagnostics

# Feature spaces; no additional backbone inference.
python -m scripts.embedding_pca
python -m scripts.cluster_agreement

# New inference for shifts > 0 m, but NO training.
python -m scripts.registration_sensitivity
```

Useful options:

```bash
python -m scripts.benchmark_comparison --input-dir outputs/embeddings --output-dir outputs/analysis/frozen_encoders
python -m scripts.embedding_pca --models resnet50 vit_b16
python -m scripts.cluster_agreement --models resnet50 vit_b16 --max-clusters 3
python -m scripts.registration_sensitivity --models resnet50 --shifts 0 2
```

The translation experiment recreates cropped images; it **does not realign surveys**. Clustering operates on the **original features**, not PCA coordinates.

### 9.5 Contrastive ResNet50: audit, training, diagnostics, ablation

**Run the audit before considering training.** Audit and diagnostic commands do not replace the historical experiments already recorded.

```bash
# Inspect folds, embargo and eligible negatives; no model is loaded.
python -m scripts.resnet50_sonar_vs_png --audit-only

# BN-affine training, three folds: WRITES checkpoints and results.
python -m scripts.resnet50_sonar_vs_png \
  --folds 3 --buffer-m 25 --epochs 3 \
  --batch-size 4 --lr 1e-5 --seed 42 --device mps

# Post-training diagnostics: require the six BN-affine checkpoints.
python -m scripts.resnet50_sonar_vs_png --diagnose-only --device mps

# Separate last-bottleneck follow-up, constrained to the BN-affine protocol.
python -m scripts.resnet50_sonar_vs_png \
  --ablation-last-bottleneck --folds 3 --epochs 3 \
  --batch-size 4 --lr 1e-5 --seed 42 --device mps
```

The [CLI](scripts/resnet50_sonar_vs_png.py) also accepts `--device cpu`, `auto` and `cuda`; `mps` targets Apple Silicon. Modes `--audit-only`, `--diagnose-only` and `--ablation-last-bottleneck` are mutually exclusive. **Do not accidentally overwrite historical checkpoints or results.** The follow-up uses `new_results/last_bottleneck/`, while standard training writes to its original location. The ablation checks saved-protocol compatibility and refuses inappropriate reuse.

#### Fine-tuning figures without retraining

```bash
python -m scripts.generate_resnet50_chapter4_figures
```

[`scripts/generate_resnet50_chapter4_figures.py`](scripts/generate_resnet50_chapter4_figures.py) reads `heldout_summary.csv` and `training_history.csv` from BN-affine and last-bottleneck folders. It generates **two PNGs** under `outputs/analysis/resnet50_cross_survey_cv/chapter4_figures/`:

- `heldout_recall_comparison.png`: frozen/BN-affine/last-bottleneck Recall@1 and Recall@3, SONAR/PNG, and chance baselines;
- `training_loss_comparison.png`: InfoNCE loss by fold and epoch.

The architecture diagram used in the report is a **separate conceptual illustration**: the illustrated patches are not documented BlueBoat samples, and this script does not generate that diagram.

### 9.6 Quick checks

```bash
# Standard-library tests:
python -m unittest discover -s tests -p 'test_*.py' -q

# Optional pytest runner:
python -m pip install pytest
python -m pytest -q

# Check that cached experimental results are present:
ls outputs/analysis/frozen_encoders/four_encoder_metrics_summary.csv
ls outputs/analysis/resnet50_cross_survey_cv/heldout_summary.csv
ls outputs/analysis/resnet50_cross_survey_cv/new_results/last_bottleneck/heldout_summary.csv

# Regenerate Chapter 4 graphics without retraining:
python -m scripts.generate_resnet50_chapter4_figures
```

These steps avoid downloading large XTF files or repeating training if only figures or reports need regeneration.

## 10. Output files, provenance, and maintenance

### 10.1 Result locations

| Analysis | Responsible code | Main output |
| --- | --- | --- |
| Dataset and manifest | [`src/data/dataset_builder.py`](src/data/dataset_builder.py) | `outputs/manifest.json`, `outputs/tables`, `outputs/patches` |
| Coverage audit | [`scripts/audit_dataset.py`](scripts/audit_dataset.py) | `outputs/tables/positive_pair_quality.csv` |
| Per-model frozen embeddings | [`src/benchmark/pipeline.py`](src/benchmark/pipeline.py) | `outputs/embeddings` |
| Frozen comparison | [`scripts/benchmark_comparison.py`](scripts/benchmark_comparison.py) | `outputs/analysis/frozen_encoders/four_encoder_metrics_summary.csv`, PNGs |
| Shared retrieval failures | [`scripts/benchmark_comparison.py`](scripts/benchmark_comparison.py) | `outputs/analysis/frozen_encoders/diagnostics/four_encoder_shared_failure_queries.csv` |
| Nominal-ID audit | [`scripts/positive_pair_diagnostics.py`](scripts/positive_pair_diagnostics.py) | `outputs/analysis/positive_pair_diagnostics` |
| Acquisition/view geometry | [`scripts/acquisition_condition_analysis.py`](scripts/acquisition_condition_analysis.py) | `outputs/analysis/acquisition_conditions` |
| PCA | [`scripts/embedding_pca.py`](scripts/embedding_pca.py) | `outputs/analysis/final_evaluation/embeddings/pca/pca_variance.csv` |
| Clustering | [`scripts/cluster_agreement.py`](scripts/cluster_agreement.py) | `outputs/analysis/final_evaluation/embeddings/clustering/cluster_agreement.csv` |
| Controlled shifts | [`scripts/registration_sensitivity.py`](scripts/registration_sensitivity.py) | `outputs/analysis/final_evaluation/registration/registration_summary.csv`, `registration_coverage.csv` |
| BN-affine fine-tuning | [`scripts/resnet50_sonar_vs_png.py`](scripts/resnet50_sonar_vs_png.py) | `outputs/analysis/resnet50_cross_survey_cv/heldout_summary.csv`, `training_history.csv` |
| Last-bottleneck fine-tuning | [`src/experiments/resnet50_sonar_png/ablation.py`](src/experiments/resnet50_sonar_png/ablation.py) | `outputs/analysis/resnet50_cross_survey_cv/new_results/last_bottleneck/heldout_summary.csv` |

Training `train_png/` directories, `.pt` checkpoints, `.npz` embeddings and `.csv` tables are derived artifacts. **`.gitignore` does not untrack files that were already committed**: always check `git status`, Git LFS tracking and sharing permissions.

### 10.2 Experimental identity and validity

For reproducibility, results should be associated with `(survey pair, patch ID)`, the `SONAR/MASK-ONLY` condition, encoder and checkpoint version, gallery size, `numerical/PNG` modality, fold/split, seed, epochs, batch size, hyperparameters, embargo specification and reported statistic. Geographic-CV metadata are recorded in `outputs/analysis/resnet50_cross_survey_cv/protocol.json` and the `outputs/analysis/resnet50_cross_survey_cv/new_results/last_bottleneck/protocol.json`.

Raw XTF data are not included; some derived SONAR arrays may contain coordinates and acquisition information. **Do not make XTFs, patches or checkpoints public or upload them without permission.** Git LFS is not a backup system and does not provide unlimited storage guarantees.

### 10.3 Automated tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v

# Optional alternative (install pytest first):
python -m pip install pytest
python -m pytest -q
```

Targeted suites include [`test_encoder_preprocessing.py`](tests/test_encoder_preprocessing.py), [`test_shared_benchmark_pipeline.py`](tests/test_shared_benchmark_pipeline.py), [`test_embedding_pca.py`](tests/test_embedding_pca.py), [`test_cluster_agreement.py`](tests/test_cluster_agreement.py), [`test_registration_sensitivity.py`](tests/test_registration_sensitivity.py), [`test_resnet50_sonar_png.py`](tests/test_resnet50_sonar_png.py), and [`test_resnet50_last_bottleneck_ablation.py`](tests/test_resnet50_last_bottleneck_ablation.py). Synthetic tests check **code correctness**, not the physical identity of SONAR matches.

### 10.4 Troubleshooting

- **Module not found:** run `python -m scripts.module_name` from the **repository root**.
- **Missing XTF or `configs/local.yaml`:** required for `build_dataset` but not for most cached-result analyses. Verify that `dataset_root` points to the parent of `SSS-XTF/`.
- **ImageNet or CLIP models unavailable:** initial use may require downloading compatible weights and dependencies; do not arbitrarily replace CLIP QuickGELU.
- **High cosine but low Recall:** relative gallery ranking and tie handling matter more than the pair's isolated cosine.
- **50 queries versus 48 queries:** these are different gallery protocols. Do not treat 48/50 as a performance change.
- **Too few examples in a fold:** the embargo may remove pairs; do not relax the buffer just to produce a preferred result.
- **Few epochs and irregular loss:** lower training loss is not evidence of generalization.
- **Versioned data/results:** `outputs/` also contains previously tracked files. Check provenance and pending changes before cleaning outputs or rerunning training.
- **Removed Markdown directories:** Git does not preserve empty directories; scripts recreate necessary output directories.

## 11. Limitations, interpretation, and future work

**Conclusions supported by the data:** the frozen benchmark does not demonstrate robust cross-survey recognition; mask-only cues and coverage quality are potential confounders; PCA and ARI/NMI provide global descriptions but do not establish correct individual matches; the finite-crop sensitivity experiment jointly changes shift and available information; contrastive BN-affine and last-bottleneck fine-tuning **do not improve Recall@1** on the evaluated geographic folds. Numerical SONAR and 8-bit PNG perform nearly identically because they encode the same intensities.

**Conclusions not supported:** underwater localization accuracy, actual GPS error, acoustically verified positive identity, a unique explanation of failures, or statistically generalizable superiority/inferiority of an architecture across all sonar data. The observations and comparisons are correlated, the dataset contains only 25 **nominal** pairs, and the last-bottleneck ablation does not use a newly acquired independent test set.

**Reasonable next steps:** collect more genuinely independent survey passes; verify cross-survey registration using external physical/acoustic references; improve valid coverage; explicitly control NoData-mask effects; only then compare domain adaptation or SSS-specific architectures on a new held-out benchmark.

---

**Maintenance:** this `README.md` centralizes the project's Markdown documentation. Update it **together with** changes to code, dataset, protocols, results and outputs. For current numerical values and configuration details, consult the linked source code, CSVs and `protocol.json` files.

