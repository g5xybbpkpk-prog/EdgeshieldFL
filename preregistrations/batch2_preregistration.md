# Pre-Registration Document: EdgeShield-FL GPU Batch 2 (Path B Completion)

**Campaign:** EdgeShield-FL GPU Batch 2  
**Date:** October 8, 2026  
**Execution Target:** 4 Arms (Batch 2A) + 1 Script (Batch 2B) + 2 Noise Arms (~7 GPU-Hours)  
**Status:** Pre-Registered Prior to Execution  

---

## 1. Executive Summary & Purpose

This pre-registration document locks the hypotheses, theoretical derivations, discriminating bands, and experimental protocols for the remaining items in **Path B**:

1. **Batch 2A — Adaptive Collusion $\epsilon$-Sweep (~4 Arms, ~4 GPU-h):**  
   Directly addresses the "strawman mimicry" objection by evaluating colluding adversaries who inject Gaussian perturbation noise $\eta_i \sim \mathcal{N}(0, \sigma_{\eta}^2 I_d)$ into their updates to evade the Tightness Tracker ($\tau_{\text{tight}} = 0.72$). We evaluate $\epsilon \in \{0.01, 0.05, 0.10, 0.20\}$ at $\alpha = 1.0$, Seed 42 ($T = 100$). This converts heuristic criticism into a quantified, closed-form evasion envelope.
2. **Batch 2B — Privacy Leakage Quantification (~1 GPU-h):**  
   Evaluates empirical information leakage from the $292$-byte inspection channel ($64$-dim sketch $\hat{s}_i$, layer profile $\hat{P}_i$, scalar norm $\hat{M}_i$). We implement a Ridge Regression linear reconstruction attack against true gradient coordinates using the public Rademacher matrix $\Phi$, plus class-distribution attribute inference from layer-norm profiles.
3. **Within-Seed Stochastic Noise Check (~2 GPU-h):**  
   Re-runs the $\alpha=1.0$ Seed 42 EdgeShield EMA sign-flip arm twice with identical Dirichlet client data partitions but independent SGD mini-batch shuffle seeds, bounding run-to-run stochastic noise.

---

## 2. Batch 2A: Adaptive Collusion $\epsilon$-Sweep

### 2.1 The Threat Model & The "Strawman" Objection
In Section 3 of the manuscript, the **Tightness Tracker** achieved $100.0\%$ True Positive Rate ($1188/1188$ rounds across 3 seeds) against exact-cloning mimics who upload identical updates $g_{\text{mal}} = \Delta w_{t-1}$.  
*The Adversarial Counter-Hypothesis:* A sophisticated colluding cohort would not submit identical clones. To evade the pairwise threshold $\tau_{\text{tight}} = 0.72$, adversaries inject independent zero-mean Gaussian noise $\eta_i$ into their updates:
$$g_i = g_{\text{mal}} + \eta_i, \quad \eta_i \sim \mathcal{N}(0, \sigma_{\eta}^2 I_d)$$

### 2.2 Mathematical Envelope & Evasion Threshold Derivation
Under streaming Rademacher projection matrix $\Phi \in \mathbb{R}^{m \times d}$ ($\Phi_{k,l} \in \{-1/\sqrt{m}, +1/\sqrt{m}\}$), the projected sketch is:
$$s_i = \Phi g_i = \Phi g_{\text{mal}} + \Phi \eta_i$$
To parameterize the noise such that each sketch dimension receives perturbation variance $\epsilon^2$, the parameter-space noise variance is calibrated as:
$$\sigma_{\eta}^2 = \epsilon^2 \cdot \frac{m}{d}$$
where $m = 64$ and $d = 136{,}874$. The expected parameter perturbation norm satisfies $\mathbb{E}[\|\eta_i\|^2] = d \sigma_{\eta}^2 = m \epsilon^2$.

With independent client-side sketch perturbation $\xi_i \sim \mathcal{N}(0, \sigma_s^2 I_m)$ ($\sigma_s = 0.05$), the uploaded sketch is $\hat{s}_i = s_i + \xi_i$. The cross-product between distinct colluding updates $i \ne j$ satisfies:
$$\mathbb{E}[\langle \hat{s}_i, \hat{s}_j \rangle] = \|g_{\text{mal}}\|^2$$
$$\mathbb{E}[\|\hat{s}_i\|^2] = \|g_{\text{mal}}\|^2 + m(\sigma_s^2 + \epsilon^2)$$

Applying Theorem 1's Delta-method expansion, the expected pairwise sketch cosine between colluding adversaries is governed by:
$$\mathbb{E}[\cos(\hat{s}_i, \hat{s}_j)] \approx \frac{\|g_{\text{mal}}\|^2}{\|g_{\text{mal}}\|^2 + m(\sigma_s^2 + \epsilon^2)}$$

At operating norm $\|g_{\text{mal}}\|^2 \approx 2.475$ ($C=5.0$) and $m = 64$, $\sigma_s = 0.05$ ($m \sigma_s^2 = 0.16$):
* **$\epsilon = 0.00$ (Exact Mimicry):** $\mathbb{E}[\cos] \approx \frac{2.475}{2.475 + 0.16} = \mathbf{0.9393}$ (Measured: $0.9364$, Caught $100\%$)
* **$\epsilon = 0.01$:** $\mathbb{E}[\cos] \approx \frac{2.475}{2.475 + 64(0.0025 + 0.0001)} = \frac{2.475}{2.641} = \mathbf{0.9370}$
* **$\epsilon = 0.05$:** $\mathbb{E}[\cos] \approx \frac{2.475}{2.475 + 64(0.0025 + 0.0025)} = \frac{2.475}{2.795} = \mathbf{0.8855}$
* **$\epsilon = 0.10$:** $\mathbb{E}[\cos] \approx \frac{2.475}{2.475 + 64(0.0025 + 0.0100)} = \frac{2.475}{3.275} = \mathbf{0.7557}$
* **$\epsilon = 0.20$:** $\mathbb{E}[\cos] \approx \frac{2.475}{2.475 + 64(0.0025 + 0.0400)} = \frac{2.475}{5.195} = \mathbf{0.4764}$

**The Critical Evasion Threshold ($\epsilon^*$):**
Setting $\mathbb{E}[\cos] = \tau_{\text{tight}} = 0.72$:
$$\epsilon^* = \sqrt{\frac{0.28 \|g\|^2}{0.72 m} - \sigma_s^2} = \sqrt{\frac{0.28 \times 2.475}{0.72 \times 64} - 0.0025} = \sqrt{0.01504 - 0.0025} \approx \mathbf{0.112}$$

### 2.3 Pre-Registered Hypotheses & Discriminating Bands

| Arm Name | Noise Scale $\epsilon$ | Predicted Pair Cosine $\mathbb{E}[\cos]$ | Tracker Status ($\tau_{\text{tight}} = 0.72$) | Pre-Registered Outcome & Band |
|:---|:---:|:---:|:---:|:---|
| `AdaptiveCollusion_eps0.01` | $\epsilon = 0.01$ | $\approx 0.937$ | **Flagged ($\gg 0.72$)** | **Detection Band:** Attacker eviction rate $\ge 95\%$; clean accuracy matches unattacked anchor ($\approx 61.5\%$) |
| `AdaptiveCollusion_eps0.05` | $\epsilon = 0.05$ | $\approx 0.886$ | **Flagged ($\gg 0.72$)** | **Detection Band:** Attacker eviction rate $\ge 90\%$; clean accuracy matches unattacked anchor |
| `AdaptiveCollusion_eps0.10` | $\epsilon = 0.10$ | $\approx 0.756$ | **Transition ($\approx 0.72$)** | **Boundary Band:** Partial eviction ($40\text{--}80\%$ rounds flagged); near-threshold scatter |
| `AdaptiveCollusion_eps0.20` | $\epsilon = 0.20$ | $\approx 0.476$ | **Evaded ($\ll 0.72$)** | **Evasion Band:** Attacker eviction drops $< 10\%$; adversaries enter aggregation unevicted |

### 2.4 The Evasion-vs-Potency Trade-off (The Collusion Dilemma)
When adversaries successfully evade the Tightness Tracker at $\epsilon = 0.20$:
* **Hypothesis H-ADAPT-1 (Attack Neutralization via Dispersion):**  
  To push pairwise cosine below $0.72$, the injected noise norm satisfies $\|\eta_i\| \approx \sqrt{64 \times 0.04} = 1.60$, which equals $102\%$ of the update norm ($\|g_{\text{mal}}\| \approx 1.57$). Adding $>100\%$ independent noise destroys the coherent gradient direction of the colluders, rendering their collective injection equivalent to zero-mean random noise that cancels upon aggregation ($\frac{1}{4}\sum_{i=1}^4 \eta_i$).  
  *Prediction:* Evasion does *not* collapse the global model; clean accuracy remains within $\pm 2.5$ points of the 16-client clean baseline ($59.0\%\text{--}62.0\%$).
* **Hypothesis H-ADAPT-2 (Temporal CUSUM Secondary Capture):**  
  Unevicted noisy updates possess perturbed trajectories. Because the noise is isotropic, updates anti-align with global descent momentum, increasing their probability of being captured by temporal directional CUSUM if their drift exceeds $\nu = 0.01$.

* **Hypothesis H-ADAPT-3 (Economic Neutralization of Mimicry):**  
  Evading colluders submit $g_i = \Delta w_{t-1} + \eta_i$, which is descent-aligned (echoing the previous broadcast direction). Unlike sign-flippers, evading mimics do not pull the model in an adversarial direction even when admitted.  
  *Pre-Registered Outcome:* Evading adversaries ($\epsilon \ge 0.2$) are admitted into aggregation but contribute near-zero malicious mass; the attack is neutralized not by detection, but by the attack's own structural economics. The $\epsilon$-sweep's scientific payload is the validated evasion envelope ($\mathbb{E}[\cos]$ vs. $\epsilon$), establishing the exact boundary of sketch-domain collusion tracking. Adversaries seeking to maintain attack potency while evading detection must inject *directed* perturbations (e.g., aligning noise with a backdoor trigger subspace), which defines an adjacent attack frontier for future research.

---

## 3. Batch 2B: Privacy Leakage Quantification

### 3.1 Scope and Information-Theoretic Identity
Address the concern that the $292$-byte inspection payload leaks proprietary client data. We quantitatively measure information leakage using two canonical adversarial inference attacks:

1. **Linear Gradient Reconstruction Attack (Ridge Regression):**
   * The server possesses public Rademacher matrix $\Phi \in \{-1/\sqrt{m}, 1/\sqrt{m}\}^{64 \times 136874}$ and noisy sketch $\hat{s}_i = \Phi g_i + \xi_i$.
   * We solve the minimum-norm ridge reconstruction:
     $$\hat{g}_{\text{recon}} = \Phi^T (\Phi \Phi^T + \lambda I)^{-1} \hat{s}_i$$
   * **Information-Theoretic Pythagorean Decomposition:**  
     Because $m = 64 \ll d = 136{,}874$, the reconstruction operator projects exclusively onto the $64$-dimensional row-space of $\Phi$: $\mathcal{P}_{\Phi}(g_i)$. The orthogonal null-space component $g_i^{\perp} = g_i - \mathcal{P}_{\Phi}(g_i)$ has dimension $d - m = 136{,}810$ and is completely unobservable. By the Pythagorean theorem:
     $$\text{NMSE} = \frac{\|g_i - \hat{g}_{\text{recon}}\|^2}{\|g_i\|^2} = 1 - \frac{\|\mathcal{P}_{\Phi}(g_i)\|^2}{\|g_i\|^2} + \frac{\|\Phi^T (\Phi \Phi^T)^{-1} \xi_i\|^2}{\|g_i\|^2} \approx 1 - \frac{m}{d} \approx \mathbf{0.9995}$$
     The attacker recovers only the $64$-dimensional projection they already observed in sketch space, learning virtually zero information about individual parameter coordinates.
   * *Pre-Registered Band:* Across $\lambda \in [10^{-4}, 10.0]$:
     $$\text{NMSE} \ge 0.95, \quad r(g_i, \hat{g}_{\text{recon}}) \le 0.15, \quad \text{Sign Agreement} \in [50.0\%, 55.0\%]$$
     (Chance sign agreement is $50.0\%$, while cleartext DLG achieves $\text{NMSE} = 0.0, r = 1.0, \text{Sign Agreement} = 100.0\%$).

2. **Attribute Inference Attack (Class Distribution Recovery from Layer Norms):**
   * Evaluated on *real* local model updates from all 20 clients across training rounds. For each client $i$, we compute true layer-norm profiles $\hat{P}_i = [\|g_{\text{conv1}}\|, \|g_{\text{conv2}}\|, \|g_{\text{fc1}}\|, \|g_{\text{fc2}}\|] + \mathcal{N}(0, \sigma_{\text{prof}}^2 I_4)$.
   * We evaluate a linear mapping $W: \mathbb{R}^4 \to \mathbb{R}^{10}$ to predict the client's true 10-class Dirichlet label distribution using **Leave-One-Out Cross-Validation (LOO-CV)** (fitting on 19 clients and evaluating on the held-out client).
   * Evaluated against two reference anchors:
     (a) *Profile-free prior guessing:* Mean class distribution of the population.
     (b) *Cleartext gradient upper bound:* Full classification-head gradients leak exact class frequencies ($\nabla_{W_c} L \propto n_c$). Measured directly via LOO-CV regression on the 10-dimensional per-class FC2 weight gradient norms, achieving near-zero TVD error ($\text{TVD} \approx 0.05$).
   * *Pre-Registered Band:* The 4 scalar layer norms provide minimal capacity to separate 10 non-IID class proportions: $\text{TVD}_{\text{LOO}} \ge \text{TVD}_{\text{prior}} - 0.05$.

---

## 4. Within-Seed Stochastic Noise Check

### 4.1 Scope and Configuration
* **Target Arm:** `EdgeShield_EMA_SignFlip20_a1.0` (Seed 42).
* **Execution:** Re-run $2\times$ (Seeds 42-run2, 42-run3) with **identical Dirichlet client data partitions** (Seed 42), but independent stochastic draws:
  * Local SGD mini-batch shuffle order perturbed via `shuffle_offset` ($10000$ and $20000$).
  * Client-side noise generators (`c_gen`, Laplace norm) include `shuffle_offset`.
  * Arm-level RNG initialized with `torch.manual_seed(seed + shuffle_offset)`.
* **Metric:** Accuracy standard deviation $\sigma_{\text{SGD}}$ across identical partitions.
* **Pre-Registered Prediction:** Stochastic mini-batch variance is bounded: $\sigma_{\text{SGD}} \le 0.50$ percentage points, confirming that the measured standard deviations ($\sigma \approx 1.6\%$ at $\alpha=1.0$) reflect Dirichlet partition variance rather than mini-batch sampling noise.

---

## 5. Master Pre-Registration Sign-Off

All bands, thresholds, and hypotheses are locked as of October 8, 2026, prior to executing Batch 2A and 2B.
