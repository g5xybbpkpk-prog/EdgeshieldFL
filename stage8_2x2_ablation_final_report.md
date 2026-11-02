# Stage 8: 2×2 Factorial Ablation Final Report
## Causal Isolation of Sketch/CUSUM Momentum Tracking vs. Server-Side Guarded Pruning Under Zero Test-Set Overlap

**Protocol Reference:** `PREREG-STAGE-8-2X2-ABLATION`  
**Protocol Hash (SHA-256):** `3eeca778616da13429aa724a0fef05c52e40ea6283a5ae159bc0bc7d7a9e45ce`  
**Primary Result Artifact:** `stage8_2x2_ablation_results.json`  
**Result Artifact Hash (SHA-256):** `975ada2ca9ba5c059dd4f24102af1a9260bffb7045f4c950d192980eea077a7f`  
**Pruning Telemetry Log:** `rollback_stage8.log`  
**Telemetry Log Hash (SHA-256):** `12c1c196d4fd853e24746c189eda447990270943c70a66864ba1478179b55caa`  
**Execution Hardware:** NVIDIA Tesla T4 GPU (executed across two sessions via atomic checkpoint resumption into `stage8_2x2_ablation_results.json` and `checkpoints/`)  
**Status:** **Milestone 1 100% COMPLETE & BALANCED** ($N=6$ seeds per cell, 12 arms total, 12 terminal checkpoints preserved)

---

## 1. Executive Summary & Core Scientific Conclusion

Stage 8 addresses a key architectural question left open by earlier composite runs:
> *Does EdgeShield-FL's $+3.26$ percentage point accuracy recovery under 20% Byzantine sign-flip attack come from temporal sketch/CUSUM momentum tracking (Pillars I–III), or from server-side activation fine-pruning (Pillar IV) gated on a server proxy set?*

The completion of the balanced Stage 8 $2 \times 2$ factorial experiment delivers a clear answer at $N = 6$ ($p = 0.012$):

1. **Pillars I–III Together Account for the Recovery Under Secure Aggregation ($\bar{D}_{\text{sketch}} = \mathbf{+3.39 \pm 2.15\%}$):**  
   Factorial Cell (1, 0) (**EdgeShield Sketch-Only**, with Pillar IV pruning completely disabled: $K_{\text{prune}} = 999$) achieves a statistically significant mean paired recovery of $+3.39\%$ over same-seed baselines from the earlier harness ($t = 3.861$, df = 5, two-sided $p = \mathbf{0.0119}$, $95\%$ CI: $\mathbf{[+1.13\%, +5.65\%]}$).
2. **Zero Test-Set Overlap Confirmed:**  
   Because pruning was disabled, the server never evaluated or gated on proxy samples. Both the full 10,000-sample test set and the disjoint 9,500-sample test set were evaluated independently with 0% data leakage, yielding consistent mean performance ($61.42\%$ full vs. $61.52\%$ disjoint, within about $0.1$ percentage points).
3. **Guarded Pruning Alone Provides Zero Recovery ($\bar{D}_{\text{prune}} = \mathbf{-1.41 \pm 1.96\%}$):**  
   In Factorial Cell (0, 1) (**FedAvg + Pruning-Only**, with sketch tracking disabled), proxy pruning on a 500-sample clean training proxy set fails to recover accuracy ($t = -1.761$, df = 5, two-sided $p = 0.1385$, $95\%$ CI: $[-3.48\%, +0.65\%]$). Without temporal momentum tracking to filter adversarial updates during aggregation, the global model remains poisoned, and activation pruning cannot reverse the Byzantine corruption.
4. **Pre-Registered Decision Outcome:**  
   Under the pre-registered decision rules of `PREREG-STAGE-8-2X2-ABLATION` ($\bar{D}_{\text{sketch}} \ge +2.0\%$ with $95\%$ CI strictly above zero, and $\bar{D}_{\text{prune}} \le +1.0\%$), **BRANCH 1 IS FORMALLY CONFIRMED**.
   > **Binding Interpretation:** Pillars I–III together account for the recovery under cryptographic secure aggregation (separating the sketch tracking pipeline from server-side pruning, without isolating individual components within Pillars I–III). Pillar IV was not shown to be necessary for Byzantine recovery; its effect when combined with sketch tracking was not tested.

---

## 2. Complete $2 \times 2$ Factorial Results Ledger

### Table A: Balanced $2 \times 2$ Factorial Cell Comparison ($N=6$ Seeds: 42, 43, 44, 45, 46, 47)

| Factorial Cell | Factor A: Sketch/CUSUM | Factor B: Proxy Pruning | Proxy Origin | Mean Test Acc (Full 10k) | Mean Test Acc (Disjoint 9.5k) | Mean Paired Recovery $\bar{D}$ | SE ($s/\sqrt{6}$) | Paired $t$-stat (df=5) | Two-Sided $p$-value | 95% Confidence Interval |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cell (0, 0)**: FedAvg Baseline | OFF | OFF | None | 58.02% | — [Not measured] | $\pm 0.00\%$ | — | — | — | [Anchor Baseline] |
| **Cell (0, 1)**: Prune-Only | OFF | ON ($K=25$) | Clean Train (0..499) | 56.61% | 56.69% | **-1.41%** | 0.80% | -1.761 | 0.1385 | [-3.48%, +0.65%] |
| **Cell (1, 0)**: Sketch-Only | ON | OFF ($K=999$) | None | 61.42% | 61.52% | **+3.39%** | 0.88% | **3.861** | **0.0119** | **[+1.13%, +5.65%]** |
| *Stage 7 Reference: Composite* | *ON* | *ON ($K=25$)* | *Test (0..499)* | *61.28%* | *— [Not measured]* | *+3.26%* | *0.60%* | *5.455* | *0.0028* | *[+1.72%, +4.80%]* |

---

### Table B: Complete Per-Seed Ledger Across All 12 Experimental Arms

#### Factorial Cell (1, 0): EdgeShield Sketch-Only (Pruning Completely OFF, $K_{\text{prune}}=999$)
*Zero proxy involvement, zero test-set leakage, 100% clean 10,000-image evaluation.*

| Seed | Undefended Baseline (FedAvg-SF) | Final Test Acc (Full 10k) | Final Test Acc (Disjoint 9.5k) | Paired Recovery $D_{\text{sketch}}$ | Attacker Flooring Rate (Rounds 30–60) | Honest Friendly Fire Rate (Rounds 1–100) | Checkpoint Path |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **42** | 59.22% | **63.16%** | 63.15% | **+3.94%** | 70.97% | 0.81% | `checkpoints/stage8_EdgeShield_NoPrune_SignFlip20_a1.0_s42_r100.pt` |
| **43** | 57.88% | **61.33%** | 61.62% | **+3.45%** | 92.74% | 3.94% | `checkpoints/stage8_EdgeShield_NoPrune_SignFlip20_a1.0_s43_r100.pt` |
| **44** | 58.11% | **57.58%** | 57.71% | **-0.53%** | 49.19% | 1.81% | `checkpoints/stage8_EdgeShield_NoPrune_SignFlip20_a1.0_s44_r100.pt` |
| **45** | 57.69% | **63.05%** | 63.23% | **+5.36%** | 75.00% | 0.31% | `checkpoints/stage8_EdgeShield_NoPrune_SignFlip20_a1.0_s45_r100.pt` |
| **46** | 58.14% | **63.37%** | 63.31% | **+5.23%** | 82.26% | 0.62% | `checkpoints/stage8_EdgeShield_NoPrune_SignFlip20_a1.0_s46_r100.pt` |
| **47** | 57.09% | **60.00%** | 60.12% | **+2.91%** | 69.35% | 2.88% | `checkpoints/stage8_EdgeShield_NoPrune_SignFlip20_a1.0_s47_r100.pt` |
| **Mean** | **58.02%** | **61.42%** | **61.52%** | **+3.39%** | **73.25%** | **1.73%** | *6 Model Checkpoints Preserved* |

---

#### Factorial Cell (0, 1): FedAvg + Proxy Pruning-Only (Sketch/CUSUM OFF, $K_{\text{prune}}=25$)
*Evaluated on clean training proxy set (CIFAR-10 train indices 0..499). Clients drawn from 500..49999.*

| Seed | Undefended Baseline (FedAvg-SF) | Final Test Acc (Full 10k) | Final Test Acc (Disjoint 9.5k) | Paired Recovery $D_{\text{prune}}$ | Attacker Flooring Rate (Rounds 30–60) | Honest Friendly Fire Rate (Rounds 1–100) | Checkpoint Path |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **42** | 59.22% | **57.65%** | 57.83% | **-1.57%** | 0.00% | 0.00% | `checkpoints/stage8_FedAvg_PruneOnly_SignFlip20_a1.0_s42_r100.pt` |
| **43** | 57.88% | **57.27%** | 57.38% | **-0.61%** | 0.00% | 0.00% | `checkpoints/stage8_FedAvg_PruneOnly_SignFlip20_a1.0_s43_r100.pt` |
| **44** | 58.11% | **53.80%** | 53.82% | **-4.31%** | 0.00% | 0.00% | `checkpoints/stage8_FedAvg_PruneOnly_SignFlip20_a1.0_s44_r100.pt` |
| **45** | 57.69% | **59.38%** | 59.39% | **+1.69%** | 0.00% | 0.00% | `checkpoints/stage8_FedAvg_PruneOnly_SignFlip20_a1.0_s45_r100.pt` |
| **46** | 58.14% | **55.94%** | 56.08% | **-2.20%** | 0.00% | 0.00% | `checkpoints/stage8_FedAvg_PruneOnly_SignFlip20_a1.0_s46_r100.pt` |
| **47** | 57.09% | **55.61%** | 55.61% | **-1.48%** | 0.00% | 0.00% | `checkpoints/stage8_FedAvg_PruneOnly_SignFlip20_a1.0_s47_r100.pt` |
| **Mean** | **58.02%** | **56.61%** | **56.69%** | **-1.41%** | **0.00%** | **0.00%** | *6 Model Checkpoints Preserved* |

---

## 3. Pruning Telemetry Breakdown (`rollback_stage8.log`)

The Stage 8 execution logged 24 pruning decisions (4 passes per run $\times$ 6 seeds) with explicit `Round={t}` timestamps:
- **Total Pruning Passes Attempted:** 24 passes
- **Accepted Passes:** 11 passes (45.8%)
- **Rolled Back Passes:** 13 passes (54.2%)
- **Mean Attempted Proxy Accuracy Delta:** $-3.65$ percentage points
- **Mean Accepted Proxy Accuracy Delta:** $-0.18$ percentage points
- **Per-Seed Accepted Rounds:**
  - Seed 42: Accepted rounds 25, 100 (rounds 50, 75 rolled back)
  - Seed 43: Accepted rounds 25, 75 (rounds 50, 100 rolled back)
  - Seed 44: Accepted rounds 75, 100 (rounds 25, 50 rolled back)
  - Seed 45: Accepted round 25 (rounds 50, 75, 100 rolled back)
  - Seed 46: Accepted rounds 50, 100 (rounds 25, 75 rolled back)
  - Seed 47: Accepted rounds 25, 75 (rounds 50, 100 rolled back)

---

## 4. Scientific Discussion & Manuscript Integration

### Key Insights
1. **The Sketch/CUSUM Channel is the Engine of Recovery:**  
   When pruning is completely removed, EdgeShield recovers **$+3.39\%$** ($p = 0.0119$). The 95% confidence interval $[+1.13\%, +5.65\%]$ excludes zero by more than 1.1 percentage points. The recovery is powered by the CUSUM velocity filter, which drives attacker weights to $w_{\text{floor}} = 0.01$ in **$73.25\%$** of attacker opportunities during the critical rounds 30–60, while inflicting only **$1.73\%$** friendly fire on honest clients.
2. **Pruning is Ineffective Against Gradient Attacks:**  
   Under Byzantine sign-flips, adversarial gradients pull all weights in the opposite direction of the true descent vector. Pruning dormant activations on a clean proxy set merely zeros out weights of selected channels; it does not correct the inverted direction of the remaining weights. Consequently, plain FedAvg + Pruning loses $-1.41$ points on average.
3. **The Stage 8 headline does not depend on test-set overlap:**  
   The Sketch-Only arm evaluates on the full 10,000-image test set without touching a single proxy sample, achieving $+3.39\%$ recovery ($95\%$ CI: $[+1.13\%, +5.65\%]$) compared with $+3.26\%$ in the Stage 7 composite reference (a $+0.13$ percentage point difference with a paired confidence interval of roughly $\pm 3.3$ percentage points, which neither confirms nor refutes an overlap effect, while establishing that the Stage 8 headline itself is free of test-set overlap).
4. **Milestone 2 Non-Execution Formally Documented:**  
   The pre-registration protocol listed Milestone 2 (Cell (1, 1): clean-proxy composite, 6 arms) as part of the planned scope. Milestone 2 was not executed; because Branch 1's pre-registered decision rules depend solely on $\bar{D}_{\text{sketch}} \ge 2.0\%$ and $\bar{D}_{\text{prune}} \le 1.0\%$, the scientific verdict does not depend on Cell (1, 1). However, its omission is an operational deviation from the planned preregistration scope (documented in §4.5.5 of the manuscript), and no clean-proxy composite evaluation exists; the Stage 7 composite run is reported strictly as a labeled historical reference.
