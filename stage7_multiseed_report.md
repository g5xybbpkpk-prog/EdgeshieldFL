# Stage 7 Empirical Report: Multi-Seed Replication & Paired Confidence Intervals ($\alpha=1.0$)

**Protocol ID:** `PREREG-STAGE-7-MULTISEED-A1.0`  
**Evaluation Date:** October 11, 2026  
**Artifacts Ingested:**  
- `stage7_alpha1_5seeds_results.json` (`1515fb54f6748d3adb1cc005ffd76510adecfdafb56659e4b8a643d98840eb64`, 6 arms, Seeds 45 & 46)  
- `stage7_alpha1_seeds_results.json` (`1e9676cb65ad8635b514560e012f4a636e488706807d09c9642554707ddbc980`, 11 arms, Seeds 45, 46, 47 completed triplets, Seed 48 anchors)  
- Parent Anchors: `stage6B_alphasweep_results.json` (Seed 42), `stage6C_alpha1_seeds_results.json` (Seeds 43 & 44)

---

## 1. Executive Summary & Decision Rules

To evaluate whether the $\alpha=1.0$ recovery margin holds across additional independent partitions in the presence of within-partition stochastic SGD noise ($\sigma_{\text{SGD}} = 3.73\%$), we executed Stage 7 on GPU hardware across new Dirichlet partition seeds ($s \in \{45, 46, 47\}$).

### 1.1 Protocol Rules & Decision Framework

To ensure complete methodological transparency, we explicitly define the stopping, replicate, and primary-test rules governing Stage 7:

1. **Replicate Round Length & Independent Run Provenance:**  
   Every completed arm across both Stage 7 artifacts (`stage7_alpha1_5seeds_results.json` and `stage7_alpha1_seeds_results.json`) executed for **all 100 training rounds (101 evaluation rounds, rounds 0 through 100 inclusive)**. Milestone 1 was an independent full 100-round execution (`stage7_alpha1_5seeds.py`), and the Full Sweep was an independent full 100-round execution (`stage7_alpha1_multiseed.py`). Both executions ran to completion for 100 rounds, and every reported accuracy in Tables 1 and 2 is an end-of-training round-100 metric.

2. **Primary Pre-Registered Conforming Analysis (Milestone 1, $N=5$):**  
   The pre-registered protocol (`preregistrations/stage7_preregistration.md`, §2–§3) declared Milestone 1 ($N=5$ seeds total: historical seeds 42–44 plus seeds 45 and 46) evaluated under a two-sided $95\%$ confidence interval ($\Delta_{\text{CI, lower}} > 0.00\%$ via $t_{N-1, 0.025}$) and recovery ratio $\bar{R} \in [25.0\%, 55.0\%]$:
   - **Mean Paired Recovery:** $\bar{D} = \mathbf{+2.64 \pm 1.78\%}$, $\text{SE}_D = 0.80\%$ ($t(4) = 3.322$, two-tailed $p = \mathbf{0.0294} < 0.05$).
   - **Two-Sided 95% Confidence Interval:** $[+0.43\%, \; +4.85\%]$ (strictly positive lower bound).
   - **Recovery Ratio:** $\bar{R} = 37.86 \pm 24.24\% \in [25.0\%, 55.0\%]$.
   - **Decision Rule Outcome:** **Branch A is confirmed under the pre-registered two-sided CI rule** at the primary Milestone 1 checkpoint.

3. **Secondary Sensitivity Analyses (Full Sweep $N=6$ & Duplicate-Averaged):**  
   - **Pooled Full Sweep Sample ($N=6$):** $\bar{D} = \mathbf{+3.26 \pm 1.46\%}$, two-tailed $t = 5.455$, df = 5, $p = 0.0028$, $95\%$ CI: $[+1.72\%, +4.80\%]$. (One-tailed $p = 0.0014$).
   - **Duplicate-Averaged Sample ($N=6$):** $\bar{D} = \mathbf{+3.01 \pm 1.12\%}$, two-tailed $t = 6.608$, df = 5, $p = 0.0012$, $95\%$ CI: $[+1.84\%, +4.18\%]$.
   - **Confirmatory Fresh-Seed Subset Alone ($N=3$):** $\bar{D} = +3.34 \pm 1.59\%$, two-tailed $t = 3.635$, df = 2, two-tailed $p = 0.068$, $95\%$ CI: $[-0.61\%, +7.29\%]$. (One-tailed $p = 0.034$). Because the pre-registered protocol specified a two-sided $95\%$ CI, the fresh-seed sample alone is underpowered to achieve two-sided significance.

4. **Deviations from Pre-Registration Protocol:**  
   We transparently disclose five procedural deviations:
   (a) Unplanned reruns of seeds 45 and 46 in the full-sweep script rather than advancing directly to seed 47.
   (b) Truncation at $N=6$ completed seeds due to Kaggle 9-hour execution cutoff (intended Milestone 2 was $N=8$).
   (c) Missing evaluations for seeds 48 (ES arm timed out) and 49 (never initiated).
   (d) Primary hypothesis tests are reported using two-sided 95% confidence intervals as specified in the protocol.
   (e) Execution was completed across two GPU sessions using disk-backed checkpoint resumption (`results_file`).

---

## 2. Partition-by-Partition Empirical Breakdown

### Table 1: Complete $N=6$ Seed Master Ledger (`stage7_alpha1_seeds_results.json`)

| Partition Seed | FedAvg Clean Acc | FedAvg Sign-Flip Acc | Baseline Deficit ($\Delta_{\text{base}}$) | EdgeShield SF Acc | Recovered Accuracy ($D_s$) | Recovery Ratio ($R_s$) | Attacker Flooring (Rnds 30–60) | Honest Friendly Fire (Rnds 1–100) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42** | $66.94\%$ | $59.22\%$ | $7.72\%$ | $61.56\%$ | **$+2.34\%$** | $30.31\%$ | $91.13\%$ ($113/124$) | $1.44\%$ ($23/1600$) |
| **Seed 43** | $64.24\%$ | $57.88\%$ | $6.36\%$ | $59.98\%$ | **$+2.10\%$** | $33.02\%$ | $95.97\%$ ($119/124$) | $2.62\%$ ($42/1600$) |
| **Seed 44** | $66.41\%$ | $58.11\%$ | $8.30\%$ | $63.23\%$ | **$+5.12\%$** | $61.69\%$ | $52.42\%$ ($65/124$) | $0.00\%$ ($0/1600$) |
| **Seed 45** | $66.01\%$ | $57.69\%$ | $8.32\%$ | $62.61\%$ | **$+4.92\%$** | $59.13\%$ | $88.71\%$ ($110/124$) | $0.75\%$ ($12/1600$) |
| **Seed 46** | $65.59\%$ | $58.14\%$ | $7.45\%$ | $59.88\%$ | **$+1.74\%$** | $23.36\%$ | $69.35\%$ ($86/124$) | $0.56\%$ ($9/1600$) |
| **Seed 47** | $66.88\%$ | $57.09\%$ | $9.79\%$ | $60.44\%$ | **$+3.35\%$** | $34.22\%$ | $57.26\%$ ($71/124$) | $2.62\%$ ($42/1600$) |
| **$N=6$ Summary** | **$66.01 \pm 1.01\%$** | **$58.02 \pm 0.70\%$** | **$7.99 \pm 1.13\%$** | **$61.28 \pm 1.41\%$** | **$+3.26 \pm 1.46\%$** | **$40.29 \pm 16.06\%$** | **$75.81 \pm 18.66\%$** | **$1.33 \pm 1.10\%$** |

*(Seed 48 Attrition Disclosure: `FedAvg_Clean_a1.0_s48` reached $64.85\%$ and `FedAvg_SignFlip20_a1.0_s48` reached $59.86\%$, a baseline deficit of $4.99$ points, before session timeout halted the paired EdgeShield run. Because its paired counterpart was truncated, Seed 48 is excluded from paired analysis per pre-registered protocol rules).*

---

### Table 2: Milestone 1 $N=5$ Ledger (`stage7_alpha1_5seeds_results.json`)

| Partition Seed | FedAvg Clean Acc | FedAvg Sign-Flip Acc | Baseline Deficit ($\Delta_{\text{base}}$) | EdgeShield SF Acc | Recovered Accuracy ($D_s$) | Recovery Ratio ($R_s$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42** | $66.94\%$ | $59.22\%$ | $7.72\%$ | $61.56\%$ | **$+2.34\%$** | $30.31\%$ |
| **Seed 43** | $64.24\%$ | $57.88\%$ | $6.36\%$ | $59.98\%$ | **$+2.10\%$** | $33.02\%$ |
| **Seed 44** | $66.41\%$ | $58.11\%$ | $8.30\%$ | $63.23\%$ | **$+5.12\%$** | $61.69\%$ |
| **Seed 45 (M1)** | $65.43\%$ | $57.54\%$ | $7.89\%$ | $57.82\%$ | **$+0.28\%$** | $3.55\%$ |
| **Seed 46 (M1)** | $63.66\%$ | $58.13\%$ | $5.53\%$ | $61.49\%$ | **$+3.36\%$** | $60.76\%$ |
| **Milestone 1 ($N=5$)** | **$65.34 \pm 1.45\%$** | **$58.18 \pm 0.63\%$** | **$7.16 \pm 1.18\%$** | **$60.82 \pm 2.05\%$** | **$+2.64 \pm 1.78\%$** | **$37.86 \pm 24.24\%$** |

---

## 3. Replicate Spread Analysis & Optimization Landscape Geometry Hypothesis

### 3.1 FedAvg Arm Reproducibility Asymmetry & Optimization Landscape Geometry

Comparing independent training executions across Milestone 1 and Full Sweep reveals an apparent reproducibility asymmetry between undefended and defended arms:
- **`FedAvg_SignFlip20` (Attacked FedAvg):** Reproduces to within **$0.15$ points** on Seed 45 ($57.54\%$ vs. $57.69\%$) and **$0.01$ points** on Seed 46 ($58.13\%$ vs. $58.14\%$).
- **`FedAvg_Clean` (Clean FedAvg):** Drifts by **$0.58$ points** on Seed 45 ($65.43\%$ vs. $66.01\%$) and **$1.93$ points** on Seed 46 ($63.66\%$ vs. $65.59\%$).
- **`EdgeShield_EMA_SignFlip20`:** Spans **$4.79$ points** on Seed 45 ($57.82\%$ vs. $62.61\%$) and **$1.61$ points** on Seed 46 ($61.49\%$ vs. $59.88\%$).

**Optimization Landscape Geometry Hypothesis:**  
We formulate this finding not as definitive evidence of isolated CUSUM path-dependency, but as an empirical hypothesis regarding optimization landscape geometry:
1. *Flat Cancellation Equilibrium under Attack:* Under $20\%$ sign-flip attack, conflicting honest and malicious update vectors pull the model into a constrained, flat adversarial saddle/cancellation equilibrium ($\approx 57.5\%\text{--}58.1\%$). In this flat region, loss gradients approach zero in the adversarial subspace, suppressing the observable effect of stochastic mini-batch orderings across independent runs.
2. *Steep Manifolds in Clean Optimization:* Conversely, clean FedAvg optimizes along unconstrained, steeper loss manifolds toward sharp minima, where differences in mini-batch sampling trajectories compound over $100$ rounds into $0.58\text{--}1.93$ point accuracy differences.
3. *Threshold Sensitivity in EdgeShield:* EdgeShield operates on smoothed temporal projections; small shifts in early momentum can alter the exact round at which a client crosses the CUSUM threshold $h=5.0$, modulating aggregation weights and compounding downstream trajectory differences.

Because RNG seeds and mini-batch ordering were not strictly frozen between sweeps, this comparison reflects an optimization landscape hypothesis rather than a controlled like-for-like RNG ablation.

### 3.2 Confirmatory Fresh-Seed Subset Alone ($N=3$, Seeds 45–47)
$$\bar{D} = +3.34 \pm 1.59\%, \quad \text{SE}_D = 0.92\%, \quad t = 3.635, \quad \text{df} = 2, \quad \text{two-tailed } p = 0.068, \quad \text{one-tailed } p = 0.034, \quad 95\% \text{ CI: } [-0.61\%, +7.29\%]$$

### 3.3 Within-Seed Duplicate Averaging ($N=6$)
Averaging duplicate evaluations for Seed 45 (mean ES: $60.22\%$, mean $D = +2.60\%$) and Seed 46 (mean ES: $60.69\%$, mean $D = +2.55\%$) across all $N=6$ partitions:
$$\bar{D} = \mathbf{+3.01 \pm 1.12\%}, \quad \text{SE}_D = 0.46\%, \quad t = 6.608, \quad \text{two-tailed } p = 0.0012, \quad \text{one-tailed } p = 0.0006, \quad 95\% \text{ CI: } [+1.84\%, +4.18\%]$$

### 3.4 Pooled Master Sample Analysis ($N=6$, Full Sweep)
$$\bar{D} = \mathbf{+3.26 \pm 1.46\%}, \quad \text{SE}_D = 0.60\%, \quad t = 5.455, \quad \text{two-tailed } p = 0.0028, \quad \text{one-tailed } p = 0.0014, \quad 95\% \text{ CI: } [+1.72\%, +4.80\%]$$
Recovery ratio $\bar{R} = 40.29 \pm 16.06\%$ ($95\%$ CI: $[23.44\%, 57.14\%]$).

### 3.5 Mechanism Scorecard
- **Attacker Flooring (Rounds 30–60):** Mean $= 75.81 \pm 18.66\%$. Evaluated over rounds 30–60 inclusive ($124$ opportunities), misses the registered $\ge 80.0\%$ threshold on sample mean due to cross-seed variance (Seeds 42, 43, 45 achieve $88.7\%\text{--}96.0\%$, while Seeds 44 and 47 achieve $52.4\%\text{--}57.3\%$). Under rounds 31–60 ($120$ opportunities), rates are $90.83\%$, $95.83\%$, $52.50\%$.
- **Honest Friendly Fire (Rounds 1–100):** Mean $= 1.33 \pm 1.10\%$. Strictly satisfies the registered ceiling $\le 2.50\%$.

### 3.6 Pillar IV Activation Fine-Pruning Telemetry, Proxy Slice Disclosure & Ablation Qualification
Across all Stage 7 EdgeShield arms, Pillar IV server-side activation pruning was invoked every 25 rounds (rounds 25, 50, 75, 100; logged in `auxiliary/rollback_stage7_5seeds.log` and `auxiliary/rollback_stage7_multiseed.log`):
1. **Proxy Dataset Slice Provenance:** The server-side proxy partitions are static index slices of the CIFAR-10 test set: `proxy_train_subset = Subset(testset, range(300))` and `proxy_val_subset = Subset(testset, range(300, 500))`. Across all Stage 7 sign-flip runs, fine-tuning is disabled ($\text{FT}=\text{False}$); zero parameter gradient updates were computed on these $500$ samples ($5\%$ of the test set). The proxy set only profiles forward activation magnitudes and tests the rollback guard. While zero gradient training eliminates direct memorization on these samples, the rollback decision conditionally gates whether global weights are retained or reverted, exerting an indirect selection effect across all $10{,}000$ test images whose isolated magnitude on disjoint test indices ($500\text{--}9999$) was not separately evaluated; this constitutes an unmeasured benchmark overlap limitation.
2. **Event Counts & Execution Timing:** At round 100, `execute_pillar4` runs at $t=100$ *before* final model evaluation (`evaluate_model` on the 10,000-sample test set). Exactly 20 pruning events occurred in Stage 7 (8 in `rollback_stage7_5seeds.log` and 12 in `rollback_stage7_multiseed.log`, representing 4 events $\times$ 5 completed runs); 83 events occur across all five auxiliary logs in the campaign repository.
3. **Capacity Removal, Gated Proxy Shifts & Accepted Gains:** With fine-tuning disabled, pruning zeroes channel weights without gradient updates. Selected and gated against clean proxy validation, pruning can raise or lower proxy accuracy (e.g., by deactivating corrupted channels): across all 83 attempted events in the repository, proxy accuracy drops by an average of $-1.83$ points, while across the 45 accepted events, the mean proxy change is $-0.10$ points. Notably, 11 of the 45 accepted passes exhibited proxy accuracy gains (mean gain $+2.1$ points; e.g., $38.0\% \to 42.0\%$ and $53.0\% \to 57.0\%$ in early exploration, and Milestone 1 Seed 45 at round 75 in Stage 7 with $60.5\% \to 62.0\%$), while the remaining accepted passes exhibited zero or negative changes bounded strictly by the $-2.0$ pt rollback safety guard (8 boundary events with $|\Delta| = 2.0$ split 6 accepted and 2 rolled back due to floating-point precision).
4. **Per-Run Accepted Pruning Passes in Stage 7 (Mapped from Execution Log Order):**
   - Milestone 1 Seed 45: 3 of 4 accepted (rounds 25, 75, 100 accepted; round 50 rolled back), $D = +0.28\%$
   - Milestone 1 Seed 46: 0 of 4 accepted (all 4 rounds 25, 50, 75, 100 rolled back), $D = +3.36\%$
   - Full Sweep Seed 45: 2 of 4 accepted (rounds 25, 75 accepted; rounds 50, 100 rolled back), $D = +4.92\%$
   - Full Sweep Seed 46: 3 of 4 accepted (rounds 25, 50, 75 accepted; round 100 rolled back), $D = +1.74\%$
   - Full Sweep Seed 47: 1 of 4 accepted (round 25 accepted; rounds 50, 75, 100 rolled back), $D = +3.35\%$
5. **Ablation Qualification & Missing No-Pruning Ablation Limitation:** In Milestone 1 Seed 46, all 4 pruning attempts (drops of $-4.0$, $-3.5$, $-4.0$, $-3.5$) were rolled back (`Rollback=True`), completing with strictly zero units pruned while achieving $D = +3.36\%$ ($R = 60.76\%$). Comparing partitions, runs with 0–1 accepted passes averaged $\bar{D} = +3.36\%$, whereas runs with 2–3 accepted passes averaged $\bar{D} = +2.31\%$. However, with $n = 5$ total completed runs, this observational comparison cannot statistically distinguish whether pruning is irrelevant or whether pruning slightly degrades utility. Crucially, Milestone 1 Seed 45 accepted its round-100 pruning pass (proxy drop $-2.0$, a boundary event), meaning its primary pre-registered final evaluation was conducted on a newly pruned model; in contrast, round-100 pruning was rolled back across all three Full Sweep seeds (45, 46, 47). In all runs with accepted passes, earlier unit zeroings (from rounds 25, 50, or 75) persisted in model weights. Because an explicit configuration ablation disabling Pillar IV ($K_{\text{prune}} > 100$) was not executed across Stage 7 seeds, recovery cannot be causally attributed to temporal sketch tracking in isolation; recovery is strictly credited to the composite EdgeShield pipeline (sketch/CUSUM tracking plus Pillar IV, with a 500-image server proxy set), and the missing standalone ablation is documented as an experimental limitation.

---

## 4. Summary

1. Both the Milestone 1 sample ($N=5$, two-sided $p = 0.0294$) and the pooled $N=6$ sample ($\bar{D} = +3.26 \pm 1.46\%$, two-sided $p = 0.0028$) show a modest positive accuracy recovery ($\sim 3$ percentage points) for the composite EdgeShield pipeline at $\alpha = 1.0$, whereas the $N=3$ fresh-seed subset alone is underpowered for two-sided significance ($p = 0.068$).
2. Because within-partition SGD variance ($\sigma_{\text{SGD}} = 3.73\%$) is comparable to the cross-seed gain, single-run recovery trajectories exhibit noticeable variance around the positive population mean.
