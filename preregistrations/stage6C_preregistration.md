# Stage 6C Pre-Registration: Multi-Seed Confirmation of the Alpha=1.0 Recovery Door

## 1. Context & Motivation
In Stage 6B (seed 42), the near-IID regime ($\alpha = 1.0$) lifted directional separation ($d' = 0.828$), reduced honest friendly fire to $1.44\%$, floored $91.13\%$ of attackers, and achieved a recovery ratio of $R = 30.31\%$ against in-harness anchor baselines (`FedAvg_Clean` = $66.94\%$, `FedAvg_SignFlip` = $59.22\%$, `EdgeShield_EMA_SignFlip` = $61.56\%$, $\Delta_{\text{baseline}} = 7.72\%$). 

Because the numerator ($2.34$ points) is smaller than measured seed-to-seed partition variance, and because the $\alpha=0.3$ regime demonstrated substantial cross-seed dispersion, Stage 6C executes seeds 43 and 44 across all three $R$-component arms ($6$ arms total, $\approx 6.8$ GPU-hours) to confirm whether the recovery door holds across random initializations.

---

## 2. Pre-Registered Hypotheses for Alpha = 1.0 (Seeds 43 & 44)

* **Mean Recovery Ratio ($R_{\text{mean}}$):** **$R \in [0.25, 0.55]$** (mean across seeds 42, 43, 44).
* **Attacker Flooring Rate (Rounds 30–60):** **$\ge 80.0\%$** across seeds (confirming high detection probability).
* **Honest Friendly-Fire Flooring (Rounds 1–100):** **$\le 2.50\%$** across seeds (confirming minimal friendly fire).
* **Early Directional Separation ($d'$):** **$d' \ge 0.70$** in rounds 5–30 across seeds.

---

## 3. Declared Binding Interpretations

1. **If $R_{\text{mean}} \ge 0.25$ and Attacker Flooring $\ge 80\%$:**  
   The recovery door is universally established across multiple seeds. The manuscript's headline stands with empirical confidence intervals: *"Temporal consensus defense in privacy-preserving FL fails under severe non-IID conditions ($\alpha \le 0.3$), but consistently succeeds when distribution overlap satisfies $\alpha \ge 1.0$ ($R = \bar{R} \pm \sigma$)."*
2. **If $R$ straddles or falls below $0.25$ in seeds 43 or 44:**  
   The honest reporting claim is: *"While geometric mechanism indicators (early separation $d' \ge 0.70$, friendly fire $\le 2.5\%$, attacker detection $\ge 80\%$) recover robustly at $\alpha = 1.0$, the final accuracy recovery ratio exhibits seed-level sensitivity to non-IID partition noise, bounding practical guarantees."*
