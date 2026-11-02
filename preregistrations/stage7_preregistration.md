# Stage 7 Pre-Registration: 5+ Seed Replication and Paired Confidence Interval Verification of the Alpha=1.0 Recovery Door

**Protocol ID:** `PREREG-STAGE-7-MULTISEED-A1.0`  
**Registration Date:** October 11, 2026  
**Primary Artifact Target:** `stage7_alpha1_seeds_results.json`  
**Execution Target:** Kaggle / Colab GPU (NVIDIA P100 / T4, ~6.5 to 15.0 GPU-hours)  
**Parent Artifacts:** `stage6B_alphasweep_results.json` (Seed 42 anchor), `stage6C_alpha1_seeds_results.json` (Seeds 43 & 44 anchors)

---

## 1. Context & Scientific Motivation

In Stage 6B and Stage 6C, evaluation across 3 independent Dirichlet partitions ($\alpha = 1.0$) yielded:
- **Seed 42:** FedAvg Clean $66.94\%$, FedAvg SF $59.22\%$, EdgeShield $61.56\%$ ($\Delta_{\text{recov}} = +2.34\%$, $R = 30.31\%$)
- **Seed 43:** FedAvg Clean $64.24\%$, FedAvg SF $57.88\%$, EdgeShield $59.98\%$ ($\Delta_{\text{recov}} = +2.10\%$, $R = 33.02\%$)
- **Seed 44:** FedAvg Clean $66.41\%$, FedAvg SF $58.11\%$, EdgeShield $63.23\%$ ($\Delta_{\text{recov}} = +5.12\%$, $R = 61.69\%$)
- **3-Seed Summary:** Deficit $= 7.46 \pm 1.00\%$, Recovered Acc $= +3.19 \pm 1.68\%$, Recovery Ratio $R = 41.67 \pm 17.39\%$.

However, in Batch 2C (§4.5.3), within-partition stochastic SGD noise was measured at **$\sigma_{\text{SGD}} = 3.73\%$** (spread $7.19$ percentage points across identical Dirichlet client splits). Because $\sigma_{\text{SGD}}$ exceeds the cross-seed accuracy gain ($+3.19\%$), the 3-seed recovery effect cannot yet be distinguished from stochastic mini-batch path dependency with statistical confidence ($N=3$, $t_{2, 0.025} = 4.303$, 95% CI on recovery spans $[-0.98\%, +7.36\%]$).

Stage 7 expands the evaluation to $N \ge 5$ seeds (adding seeds 45 and 46 to reach $N=5$, with option to extend to seeds 47, 48, 49 for $N=8$). By evaluating matched triplets (`FedAvg_Clean`, `FedAvg_SignFlip20`, `EdgeShield_EMA_SignFlip20`) under identical partition seeds, we conduct a **within-seed paired difference analysis** that controls for inter-partition baseline variance.

---

## 2. Experimental Configurations & Scope

For each new random initialization seed $s \in \{45, 46, 47, 48, 49\}$ at $\alpha = 1.0$, three arms are executed for 100 rounds:

1. **`FedAvg_Clean_a1.0_s{s}`:**
   - Defense: `fedavg` (Cleartext)
   - Attack: `none` ($\beta = 0.0$, $f = 0$)
   - Hyperparameters: $C=5.0$, batch clip $= 5.0$, $E=2$, $B=64$, $\eta=0.08$.
   - Role: Upper performance ceiling anchor for seed $s$.

2. **`FedAvg_SignFlip20_a1.0_s{s}`:**
   - Defense: `fedavg` (Cleartext, undefended)
   - Attack: `sign_flip` ($\beta = 0.20$, $f = 4$ colluders)
   - Hyperparameters: $C=5.0$, batch clip $= 5.0$, $E=2$, $B=64$, $\eta=0.08$.
   - Role: Lower attacked baseline anchor for seed $s$.

3. **`EdgeShield_EMA_SignFlip20_a1.0_s{s}`:**
   - Defense: `edgeshield` (292-byte encrypted sketches, $m=64$)
   - Attack: `sign_flip` ($\beta = 0.20$, $f = 4$ colluders)
   - Parameters: $k=25$ fine-pruning, EMA consensus reference, $C=5.0$, batch clip $= 5.0$.
   - Role: Defense evaluation arm under attack for seed $s$.

### Execution Milestones:
- **Milestone 1 ($N=5$ Seeds Total):** Execute Seeds 45 & 46 (6 arms total, $\approx 6.5$ GPU-hours).
- **Milestone 2 ($N=8$ Seeds Total):** Execute Seeds 47, 48, 49 (9 additional arms, $\approx 9.5$ GPU-hours).

---

## 3. Pre-Registered Hypotheses & Test Statistics

Let $D_s = \text{Acc}_{\text{EdgeShield}}(s) - \text{Acc}_{\text{FedAvg-SF}}(s)$ denote the paired accuracy recovery on seed $s$.

1. **Hypothesis H-STAGE7-1 (Statistically Significant Paired Recovery):**
   The sample mean paired difference $\bar{D} = \frac{1}{N} \sum_{s} D_s$ satisfies:
   $$\text{CI}_{95\%}(\bar{D}) = \left[ \bar{D} - t_{N-1, 0.025} \cdot \frac{s_D}{\sqrt{N}}, \; \bar{D} + t_{N-1, 0.025} \cdot \frac{s_D}{\sqrt{N}} \right]$$
   where the lower bound of the 95% Confidence Interval strictly exceeds zero:
   $$\Delta_{\text{CI, lower}} > 0.00\%$$

2. **Hypothesis H-STAGE7-2 (Recovery Ratio Magnitude):**
   The pooled mean recovery ratio across all $N$ seeds satisfies:
   $$\bar{R} = \frac{1}{N} \sum_{s=1}^N \frac{D_s}{\text{Deficit}(s)} \in [25.0\%, 55.0\%]$$

3. **Hypothesis H-STAGE7-3 (High Attacker Flooring Probability):**
   The mean percentage of adversaries floored ($w_i \le w_{\text{floor}}$) across rounds 30–60 remains:
   $$\text{Flooring Rate}_{\text{att}} \ge 80.0\%$$

4. **Hypothesis H-STAGE7-4 (Low Friendly-Fire Eviction):**
   The mean percentage of honest clients evicted across rounds 1–100 remains:
   $$\text{Eviction Rate}_{\text{honest}} \le 2.50\%$$

---

## 4. Binding Decision Rules & Scientific Interpretations

To prevent post-hoc rationalization, we pre-commit to the following binding reporting rules:

### Branch A: The Recovery Door is Statistically Confirmed
* **Condition:** $\Delta_{\text{CI, lower}} > 0.00\%$ (95% CI strictly positive) AND $\bar{R} \ge 25.0\%$.
* **Binding Interpretation:**  
  The recovery door at $\alpha = 1.0$ is validated as a statistically reproducible effect surviving cross-partition variation and SGD noise. The manuscript updates Table 3, Table 4, and §4.5 with the full $N$-seed paired CI and paired $t$-test ($p < 0.05$), validating that EdgeShield provides statistically significant recovery without cleartext access.

### Branch B: The Recovery Door is Statistically Unresolved at the SGD Noise Floor
* **Condition:** $\Delta_{\text{CI, lower}} \le 0.00\%$ (95% CI spans zero) OR $\bar{R} < 25.0\%$.
* **Binding Interpretation:**  
  The recovery door is formally classified as statistically unresolved at the $\sigma_{\text{SGD}} = 3.73\%$ noise floor. The manuscript updates §4.5 and the Abstract to report:  
  *"While EdgeShield maintains geometric detection mechanisms (flooring $\ge 80\%$, friendly fire $\le 2.5\%$), the net accuracy recovery ratio at $\alpha=1.0$ ($R = \bar{R} \pm \text{CI}$) cannot be statistically separated from zero under multi-seed partition variance, confirming that practical federated deployment guarantees are bounded by stochastic gradient noise."*

### Branch C: Catastrophic Breakdown (Falsification)
* **Condition:** Attacker flooring drops below $60.0\%$ OR friendly fire exceeds $5.0\%$.
* **Binding Interpretation:**  
  The geometric consensus model fails under certain non-IID partition geometries even at $\alpha=1.0$, demonstrating reference contamination similar to $\alpha=0.3$.
