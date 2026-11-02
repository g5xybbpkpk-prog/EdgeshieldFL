# Stage 8 Pre-Registration: 2x2 Factorial Ablation of Sketch/CUSUM Momentum Tracking vs. Server-Side Guarded Pruning Under Zero Test-Set Overlap (Alpha = 1.0)

**Protocol ID:** `PREREG-STAGE-8-2X2-ABLATION`  
**Registration Date:** October 13, 2026  
**Primary Artifact Target:** `stage8_2x2_ablation_results.json`  
**Execution Target:** Kaggle / Cloud GPU (NVIDIA P100 / T4, ~11.5 to 17.5 GPU-hours)  
**Parent Artifacts:** `stage6B_alphasweep_results.json`, `stage6C_alpha1_seeds_results.json`, `stage7_alpha1_seeds_results.json`

---

## 1. Context & Scientific Motivation

Across Stages 6B, 6C, and 7, evaluation of EdgeShield-FL under Dirichlet $\alpha = 1.0$ and $20\%$ Byzantine sign-flip attack demonstrated a modest, statistically significant positive recovery effect ($\bar{D} = +3.26 \pm 1.46\%$, paired $t = 5.455$, pooled two-sided $p = 0.0028$, $N=6$ seeds).

However, rigorous forensic review revealed two fundamental methodological qualifications:
1. **Confounded Pipeline Attributions:** EdgeShield arms execute both *Pillar I–III temporal sketch/CUSUM momentum tracking* and *Pillar IV server-side guarded activation fine-pruning* ($K_{\text{prune}} = 25$). Across repository logs, 11 of 45 accepted pruning passes exhibited positive clean proxy validation gains (mean gain $+2.1$ points), establishing that pruning selected and gated on clean proxy data can remove corrupted channels and improve task accuracy. Because a standalone no-pruning ablation was not executed across Stage 7 seeds, recovery cannot be causally isolated to sketch tracking alone.
2. **Benchmark Test-Slice Overlap:** In the original experimental harness (`edgeshield_v2.py`), the 500 server proxy samples were sliced from indices $0\text{--}499$ of the CIFAR-10 *test* set. While zero gradient steps were computed in sign-flip arms ($\text{FT} = \text{False}$), the rollback decision conditionally gated whether global weights were kept or reverted across all rounds, exerting an unmeasured indirect selection effect across the 10,000-image test set.

**Stage 8 resolves both questions simultaneously via a balanced $2 \times 2$ factorial experiment with zero test-set overlap.**

---

## 2. Experimental Design: The $2 \times 2$ Factorial Grid

The experiment is conducted across Seeds 42, 43, 44, 45, 46, and 47 ($N=6$) under Dirichlet $\alpha = 1.0$, $20\%$ sign-flip attack ($f=4$ colluders out of $N=20$ clients, scaling factor $-3.0$), unconstrained update clipping ($C=5.0$, batch clip $=5.0$), $E=2$ local epochs, $B=64$, $\eta=0.08$, for $T=100$ communication rounds.

### The $2 \times 2$ Matrix:

| | **Factor B: Pillar IV Pruning OFF** ($K_{\text{prune}} > 100$) | **Factor B: Pillar IV Pruning ON** ($K_{\text{prune}} = 25$, Clean Train Proxy) |
|---|---|---|
| **Factor A: Sketch/CUSUM OFF** (Plain FedAvg) | **Cell (0, 0): Baseline FedAvg-SF**<br>*(Existing benchmark arms across seeds 42–47; mean $= 58.02 \pm 0.70\%$)* | **Cell (0, 1): FedAvg + Proxy Pruning Only**<br>*(New: Plain averaging + Pillar IV pruning on clean train proxy; 6 arms)* |
| **Factor A: Sketch/CUSUM ON** (EdgeShield) | **Cell (1, 0): EdgeShield Sketch-Only**<br>*(New: Full EdgeShield with $K_{\text{prune}} = 999$, zero pruning, zero proxy; 6 arms)* | **Cell (1, 1): EdgeShield Full Pipeline**<br>*(Clean Rerun: Full EdgeShield + Pillar IV with clean train proxy; 6 arms)* |

---

## 3. Strict Methodological Safeguards

To ensure that Stage 8 results are unassailable, the execution harness adheres to five binding procedural constraints:

1. **Zero Test-Set Overlap (Clean Training Root Set):**  
   The 500 server proxy samples are drawn strictly from the CIFAR-10 **training** split (`trainset`):
   - `proxy_train_subset = Subset(trainset, range(0, 300))` (for activation profiling)
   - `proxy_val_subset = Subset(trainset, range(300, 500))` (for rollback guarding, threshold $\Delta_{\text{clean}} \le 2.0$ pts)
   - Client non-IID partitions are formed over the remaining training samples (`range(500, 50000)`), ensuring complete mutual disjointness between client training partitions, the server proxy set, and the 10,000-sample test set.
2. **Explicit Per-Round Pruning Telemetry:**  
   Every invocation of `execute_pillar4` logs the explicit round index $t$, attempted delta, accepted/rollback status, and unit list to `rollback_stage8.log` (e.g., `Round=25`, `Round=50`, `Round=75`, `Round=100`).
3. **Dual Test Set Reporting:**  
   At every evaluation round, model accuracy is evaluated and recorded on:
   - Full 10,000-image CIFAR-10 test set (`testset`).
   - Disjoint 9,500-image test subset (`Subset(testset, range(500, 10000))`) to provide complete empirical continuity with earlier drafts.
4. **Terminal Checkpoint Preservation:**  
   The round-100 global model weights (`state_dict`) for every completed arm are serialized to disk (`checkpoints/stage8_{arm_name}_r100.pt`), enabling offline verification without re-training.
5. **Atomic Checkpointing & Re-Entrant Resumption:**  
   The simulation script saves completed arm results atomically to `stage8_2x2_ablation_results.json`. In the event of a Kaggle session timeout, re-launching the script detects existing entries and resumes from the next pending arm without data loss.

---

## 4. Execution Scope & Milestone Structure

- **Milestone 1 (Primary Causal Isolation, 12 Arms, ~11.5 GPU-hours):**
  - Execute **Cell (1, 0): EdgeShield Sketch-Only** (Seeds 42–47, 6 arms).
  - Execute **Cell (0, 1): FedAvg + Proxy Pruning Only** (Seeds 42–47, 6 arms).
  - Compare directly against existing matched Cell (0, 0) FedAvg-SF and FedAvg-Clean runs.
- **Milestone 2 (Uncontaminated Full Pipeline Rerun, 6 Arms, ~5.8 GPU-hours):**
  - Execute **Cell (1, 1): EdgeShield Full Pipeline with Clean Train Proxy** (Seeds 42–47, 6 arms).

---

## 5. Pre-Registered Hypotheses & Test Statistics

For each seed $s \in \{42, 43, 44, 45, 46, 47\}$, paired recovery differences are defined against the matched undefended baseline $\text{Acc}_{\text{FedAvg-SF}}(s)$:
- $D_{\text{sketch}}(s) = \text{Acc}_{\text{ES, sketch-only}}(s) - \text{Acc}_{\text{FedAvg-SF}}(s)$
- $D_{\text{prune}}(s) = \text{Acc}_{\text{FedAvg, prune-only}}(s) - \text{Acc}_{\text{FedAvg-SF}}(s)$
- $D_{\text{full}}(s) = \text{Acc}_{\text{ES, full}}(s) - \text{Acc}_{\text{FedAvg-SF}}(s)$

All primary tests are evaluated using two-sided $95\%$ paired Confidence Intervals ($t_{5, 0.025} = 2.571$):
$$\text{CI}_{95\%}(\bar{D}) = \left[ \bar{D} - 2.571 \cdot \frac{s_D}{\sqrt{6}}, \; \bar{D} + 2.571 \cdot \frac{s_D}{\sqrt{6}} \right]$$

---

## 6. Binding Decision Rules & Scientific Interpretation

### Branch 1: Sketch/CUSUM Momentum Tracking is the Primary Causal Driver
* **Condition:** $\bar{D}_{\text{sketch}} \ge +2.0\%$ with $\text{CI}_{95\%}(\bar{D}_{\text{sketch}})$ strictly above zero ($p < 0.05$), AND $\bar{D}_{\text{prune}} \le +1.0\%$.
* **Binding Interpretation:**  
  Recovery is causally established as a property of temporal sketch tracking and CUSUM momentum filtering under cryptographic secure aggregation. Pillar IV activation fine-pruning is confirmed to be an auxiliary backdoor suppression mechanism that is unnecessary for Byzantine recovery.

### Branch 2: Server-Side Proxy Pruning is the Primary Causal Driver
* **Condition:** $\bar{D}_{\text{prune}} \ge +2.0\%$ with $\text{CI}_{95\%}(\bar{D}_{\text{prune}})$ strictly above zero ($p < 0.05$), AND $\bar{D}_{\text{sketch}} \le +1.0\%$.
* **Binding Interpretation:**  
  Recovery is causally established as a property of server-side capacity pruning gated on clean proxy data. The sketch tracking channel provides input privacy and replication collusion defense, but does not drive Byzantine accuracy recovery.

### Branch 3: Synergistic Interaction
* **Condition:** Both $\bar{D}_{\text{sketch}} > 0.0$ and $\bar{D}_{\text{prune}} > 0.0$, but neither alone accounts for full recovery, and $\bar{D}_{\text{full}} > \max(\bar{D}_{\text{sketch}}, \bar{D}_{\text{prune}})$.
* **Binding Interpretation:**  
  Both mechanisms interact synergistically: temporal CUSUM filtering suppresses adversarial gradient velocity during aggregation, while Pillar IV pruning removes residual corrupted channels post-aggregation. Both are required for optimal utility.

### Branch 4: Stochastic Indeterminacy (Underpowered at SGD Noise Floor)
* **Condition:** Neither channel achieves statistically significant separation from zero at $N=6$ due to within-partition SGD noise ($\sigma_{\text{SGD}} \approx 3.7\%$).
* **Binding Decision:**  
  Extend evaluation to Seeds 48–53 ($N=12$ seeds total) using the identical frozen script without altering decision thresholds or post-hoc reframing.
