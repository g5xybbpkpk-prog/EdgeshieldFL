# EdgeShield-FL

**Mapping the Empirical Boundaries of Privacy-Preserving Defense in Heterogeneous Federated Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Reproducibility Audit](https://img.shields.io/badge/Reproducibility-100%25%20Verified-brightgreen.svg)](#quickstart-reproducing-paper-tables--metrics)

This repository contains the official reference implementation, experimental logs, pre-specified protocol registries, round-100 PyTorch model checkpoints, and automated verification scripts for **EdgeShield-FL**.

---

## Overview

Privacy-preserving federated learning (FL) on resource-constrained edge networks faces a fundamental tension between cryptographic input privacy, Byzantine/backdoor robustness, and non-IID data heterogeneity. **EdgeShield-FL** investigates a lightweight dual-channel architecture that pairs a **292-byte obfuscated client inspection payload** (64-dimensional streaming Rademacher sketch, 4-layer norm profile, and Laplace-perturbed scalar norm) with **zero-sum masked ring aggregation** (`AES-128-CTR` over $\mathbb{Z}_{2^{16}}$).

Rather than relying on throttled gradient clipping ($C=1.0$) that artificially suppresses attack variance, this benchmark suite maps where sketch-based defenses hold and where they break down under unconstrained clipping ($C=5.0$) across Dirichlet non-IID regimes ($\alpha \in \{0.1, 0.3, 1.0\}$):

- **Exact & Adaptive Collusion Envelope (§3):** The **Tightness Tracker** detects exact-cloning colluders with **100.0% TPR** ($1,188/1,188$ rounds across 3 seeds) and **0.04% FPR** ($2/4,752$ client-rounds at $\alpha \ge 0.3$), matching Theorem 1's closed-form correlation envelope ($0.9393$ predicted vs. $0.9364$ measured). Against adaptive colluders injecting Gaussian perturbations, we validate the critical evasion threshold $\epsilon^* = 0.112$ and show that evading detection ($\epsilon = 0.20$) requires $>100\%$ noise relative to update norm, diluting attack potency ($57.58\%$ clean accuracy, $4.86\%$ ASR).
- **Null-Space Privacy Bounds (§5.3):** Minimum-norm Ridge gradient reconstruction from the 292-byte sketch is lower-bounded by the $99.953\%$ unobservable orthogonal null space ($d - m = 136,810$), yielding $\text{NMSE} = 1.0013$ (vs. $0.99953$ theory), Pearson $r = 0.010$, and $44.52\%$ sign agreement. Leave-one-out attribute inference from 4-layer norm profiles yields negative information gain ($\Delta\text{TVD} = -0.0167$).
- **Empirical Failure Boundaries (§4.1–4.4):** We characterize four structural boundaries: (1) late-phase directional separation collapse ($d': 0.57 \to 0.13$) and **self-referential reference contamination** under moderate/extreme skew ($\alpha \le 0.3$), which causes unbounded friendly-fire cascades; (2) continuous single-attacker ($f=1$) stealthy backdoor implantation ($60.60 \pm 14.81\%$ ASR at zero clean accuracy penalty); (3) pairwise subspace indistinguishability below $\beta^* \approx 0.33$; and (4) the inability of post-hoc activation fine-pruning to excise distributed backdoors under a $2.0$-point clean validation guard.
- **Heterogeneity Regime Shift & $2\times 2$ Causal Ablation (§4.5):** At near-IID overlap ($\alpha = 1.0$), a balanced **$2\times 2$ factorial ablation under zero test-set overlap** ($N=6$ seeds, 12 arms) isolates **Pillars I–III temporal sketch tracking** as the driver of Byzantine sign-flip recovery ($\bar{D}_{\text{sketch}} = \mathbf{+3.39 \pm 2.15\%}$, 95% CI $[+1.13\%, +5.65\%]$, $p = 0.0119$), whereas server-side proxy pruning on plain FedAvg yields $\bar{D}_{\text{prune}} = -1.41\%$.

---

## Quickstart: Reproducing Paper Tables & Metrics

### 1. Recompute All Manuscript Tables from Raw Logs (Zero Dependencies)

`verify_all_tables_canonical.py` recomputes every table cell, mean, standard deviation, confidence interval, $t$-statistic, $p$-value, hardware cycle/SRAM specification, and pruning log count directly from the 17 raw JSON simulation logs and runtime logs. It requires **only the Python 3.8+ standard library** (no external packages needed):

```bash
python3 verify_all_tables_canonical.py
```

### 2. Evaluate Round-100 PyTorch Checkpoints

`eval_checkpoints.py` verifies all 12 serialized round-100 model checkpoints (`checkpoints/stage8_*_r100.pt`) from the Stage 8 $2\times 2$ factorial ablation:

```bash
# Offline structural & pruning-log verification (0 dependencies)
python3 eval_checkpoints.py

# Full 10,000-image & disjoint 9,500-image CIFAR-10 test inference (PyTorch or NumPy)
python3 eval_checkpoints.py --data_dir ./data
```

### 3. Run Training Benchmarks

To install optional training dependencies and run the benchmark suites on GPU:

```bash
pip install -r requirements.txt

# Run the Stage 8 2x2 factorial ablation (zero test-set overlap)
python3 auxiliary/stage8_2x2_ablation.py

# Run harness-matched Cell (0, 0) FedAvg-SF baselines on the 49,500-image split
python3 auxiliary/run_stage8_cell00_baselines.py

# Run the main 4-pillar EdgeShield-FL benchmark suite
python3 auxiliary/edgeshield_v2.py
```

---

## Repository Structure

```text
edgeshield-fl/
├── EdgeShield_FL_Final_Paper.md         # Full paper manuscript with proofs and tables
├── verify_all_tables_canonical.py       # Automated recomputation script for all tables
├── eval_checkpoints.py                  # Standalone evaluator for round-100 checkpoints
├── requirements.txt                     # Optional Python dependencies for GPU training
├── CITATION.cff                         # Citation metadata
├── LICENSE                              # MIT License
├── checkpoints/                         # 12 round-100 PyTorch .pt checkpoints (Stage 8)
├── auxiliary/                           # Training pipelines, sensitivity scripts & logs
│   ├── edgeshield_v2.py                 # Core 4-pillar client-server training pipeline
│   ├── stage8_2x2_ablation.py           # 2x2 factorial ablation harness (0% test overlap)
│   ├── run_stage8_cell00_baselines.py   # Harness-matched Cell (0,0) baseline runner
│   ├── stage7_alpha1_multiseed.py       # Multi-seed alpha=1.0 replication harness
│   ├── stage7_alpha1_5seeds.py          # 5-seed Milestone 1 replication harness
│   ├── stage4_5_bridge.py               # Bridge validation & guarded fine-pruning script
│   ├── reconcile_cusum.py               # Phase-resolved CUSUM drift analysis utility
│   ├── sweep_all_regimes_cusum.py       # Cross-regime drift accumulation analysis
│   └── rollback*.log                    # Runtime activation pruning & rollback logs
├── preregistrations/                    # Pre-specified experimental protocols
│   ├── stage6C_preregistration.md
│   ├── batch1_preregistration.md
│   ├── batch2_preregistration.md
│   ├── stage7_preregistration.md
│   └── stage8_2x2_ablation_preregistration.md
├── stage7_multiseed_report.md           # Detailed Stage 7 multi-seed technical report
├── stage8_2x2_ablation_final_report.md  # Detailed Stage 8 2x2 ablation technical report
└── *.json                               # 17 primary experimental result logs (see ledger)
```

---

## Experimental Artifact & Checksum Ledger

All 17 experimental JSON artifacts and protocol files are indexed below with their SHA-256 checksums:

| File | Description | SHA-256 Checksum |
|:---|:---|:---|
| `stage2_sweep_results.json` | Stage 2 norm clipping sweep ($C \in \{1.0, 5.0, 10.0, \infty\}$) | `2d8e0cfdb89e346a82adc85ad0a314e62b88a33c67f72a8e33feaf9105639181` |
| `stage4_results.json` | Stage 4 attack modalities & forensics (SF-20, Backdoors A–D) | `67c0e9d977675bc677abc3efb01e8fecef8128d53ed98b580f1d658d8ded35b4` |
| `bridge_results.json` | Stage 4.5 bridge validation & fine-pruning ($k=10, 25$) | `89add395dafec2b22f64b8102775e013715afcfa65c87b0dfaebe919140637ab` |
| `bridge2_results.json` | Stage 5 EMA momentum, $f=1$ single-attacker, exact mimicry | `bad34657612ae627f0a83fc744a648208871bb6a1a591f7f3921abe1e2094daf` |
| `stage6A_seeds_results.json` | Stage 6A multi-seed replication at $\alpha=0.3$ (seeds 43, 44) | `d8d0a04a83271967c731d09183484d04f3865c195b9b9dc25b21a6f77bbe61a8` |
| `stage6B_alphasweep_results.json` | Stage 6B Dirichlet $\alpha$-sweep ($\alpha \in \{0.1, 1.0\}$) | `32d229b0892d37044e967282d1697a212e64490cf3a7329978ad9b19be7a1c31` |
| `stage6C_alpha1_seeds_results.json` | Stage 6C near-IID ($\alpha=1.0$) replication (seeds 43, 44) | `3d90d20353c255ae73517d704ff6e2c8d3539822e1083acc6dd96d6ccd7f04e7` |
| `batch1A_results.json` | Batch 1A cleartext baselines (Trimmed Mean, Coordinate Median) | `d01a76f54cef6c2880dff4bc7c1135b206a61fcf166c6d9bd6bde1dd2d573fe9` |
| `batch1BC_results.json` | Batch 1BC 3-seed Random-Evict null controls & clean anchors | `3a979f51821571c7aecacf1c36652f810b21cdea7833841c20d5f83ebaf17bb7` |
| `batch1_controls_krum_results.json` | Batch 1 clean interaction controls & Multi-Krum seed 42 anchors | `163d0c7ab3b302673d8473a0ac319f2b657a4b37fad4093b9414596a1d464337` |
| `batch1_krum_seeds_results.json` | Batch 1 Multi-Krum replication (seeds 43, 44 at $\alpha=0.3, 1.0$) | `4d7d8a288bec28843644fc5efab11a8fa93590f8d4bc8ec981ff6f151140fac1` |
| `batch2A_adaptive_results.json` | Batch 2A adaptive collusion sweep ($\epsilon \in \{0.01, 0.05, 0.10, 0.20\}$) | `c0f67be1e06748e91e1b269258981d7ce8066c65b5bd7e10a4ab139b98e3174a` |
| `batch2B_privacy_results.json` | Batch 2B linear Ridge inversion & LOO attribute inference | `e9ea4e2838d763d34a6e42cb2311413f43722072437f4c2887b703c1c4282634` |
| `batch2C_noise_check_results.json` | Batch 2C within-partition stochastic noise check (offsets 10k, 20k) | `226b281c91b1b367f5f019b85c00441520b821aaa0625d827d953f044a90cdb8` |
| `stage7_alpha1_5seeds_results.json` | Stage 7 Milestone 1 5-seed replication ($\alpha=1.0$, seeds 45, 46) | `1515fb54f6748d3adb1cc005ffd76510adecfdafb56659e4b8a643d98840eb64` |
| `stage7_alpha1_seeds_results.json` | Stage 7 full multi-seed sweep ($\alpha=1.0$, seeds 45–48) | `1e9676cb65ad8635b514560e012f4a636e488706807d09c9642554707ddbc980` |
| `stage8_2x2_ablation_results.json` | Stage 8 $2\times 2$ factorial ablation (Seeds 42–47, 12 arms, 0% overlap) | `975ada2ca9ba5c059dd4f24102af1a9260bffb7045f4c950d192980eea077a7f` |
| `verify_all_tables_canonical.py` | Automated recomputation and verification script | `e6011d19c8f9d6ecc7eee80d2c773af9c13fe47bb60aab077e9b2404e05cc383` |

---

## License

This project is released under the [MIT License](LICENSE).
