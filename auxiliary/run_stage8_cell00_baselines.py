#!/usr/bin/env python3
"""
run_stage8_cell00_baselines.py

Executes the 6 harness-matched Cell (0, 0) FedAvg Sign-Flip baseline runs
(Seeds 42, 43, 44, 45, 46, 47) inside the exact Stage 8 harness:
  - Client training pool: 49,500 images (CIFAR-10 trainset[500..49999], 0% overlap with trainset[0..499])
  - Defense mode: 'fedavg' (Sketch/CUSUM OFF)
  - Pruning: OFF (enable_pruning=False, k_prune=999)
  - Attack: 20% Byzantine sign-flip (scale = -3.0, C = 5.0, T = 100 rounds)
  - Dual evaluation: Full 10,000 test set (testset[0..9999]) and Disjoint 9,500 test set (testset[500..9999])
  - Saves round-100 checkpoints to: checkpoints/stage8_FedAvg_NoPrune_SignFlip20_a1.0_s{seed}_r100.pt
  - Saves/merges results into: stage8_cell00_matched_baselines.json (and optionally stage8_2x2_ablation_results.json)
  - Computes exact paired recovery differences D_sketch and D_prune against both:
      (a) the 49,500-image harness-matched Cell (0, 0) baselines (10k Full & 9.5k Disjoint), and
      (b) the 50,000-image Stage 6C/7 historical baselines, turning the 1% partition-offset note into a direct measurement.
"""

import os
import sys
import json
import math
import numpy as np

# Import core Stage 8 harness definitions from auxiliary/stage8_2x2_ablation.py if present,
# or run standalone with identical definitions.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "auxiliary"))

from stage8_2x2_ablation import run_arm, load_existing_results

CELL00_RESULTS_FILE = "stage8_cell00_matched_baselines.json"
STAGE8_RESULTS_FILE = "stage8_2x2_ablation_results.json"

HISTORICAL_50K_SF = {
    42: 59.22,
    43: 57.88,
    44: 58.11,
    45: 57.69,
    46: 58.14,
    47: 57.09,
}


def load_cell00_results():
    if os.path.exists(CELL00_RESULTS_FILE):
        with open(CELL00_RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_atomic_json(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def calc_paired_stats(diffs):
    n = len(diffs)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    se = std / math.sqrt(n)
    t_stat = mean / se if se > 0 else 0.0
    try:
        from scipy import stats
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    except ImportError:
        # Exact df=5 two-sided 95% t-critical fallback
        t_crit = 2.5705818356363146 if n == 6 else 2.0
        p_val = float("nan")
    ci = (mean - t_crit * se, mean + t_crit * se)
    return mean, std, se, t_stat, p_val, ci


def main():
    seeds = [42, 43, 44, 45, 46, 47]
    cell00_results = load_cell00_results()
    stage8_results = load_existing_results()

    for s in seeds:
        arm_name = f"FedAvg_NoPrune_SignFlip20_a1.0_s{s}"
        if arm_name in cell00_results:
            print(f"[SKIP] {arm_name} already completed: Full={cell00_results[arm_name]['final_clean_acc_full']:.2f}%")
            continue
        if arm_name in stage8_results:
            cell00_results[arm_name] = stage8_results[arm_name]
            save_atomic_json(CELL00_RESULTS_FILE, cell00_results)
            print(f"[IMPORTED] {arm_name} from {STAGE8_RESULTS_FILE}: Full={cell00_results[arm_name]['final_clean_acc_full']:.2f}%")
            continue

        print(f"\n>>> Launching Harness-Matched Cell (0,0) Baseline: {arm_name} (49,500-image pool)")
        res = run_arm(
            arm_name=arm_name,
            rounds=100,
            beta=0.20,
            defense_mode="fedavg",
            attack_type="sign_flip",
            seed=s,
            enable_pruning=False,
            k_prune=999,
            ft_on=False,
            ema_reference=False,
        )
        cell00_results[arm_name] = res
        save_atomic_json(CELL00_RESULTS_FILE, cell00_results)

    print("\n" + "=" * 90)
    print("HARNESS-MATCHED CELL (0, 0) BASELINE MEASUREMENT REPORT (49,500 vs. 50,000 TRAIN POOL)")
    print("=" * 90)
    print(f"{'Seed':<6} | {'50k SF (Stage 7)':<18} | {'49.5k SF (Full 10k)':<21} | {'49.5k SF (Disj 9.5k)':<21} | {'Offset Delta (Full)':<18}")
    print("-" * 90)

    offset_deltas = []
    for s in seeds:
        arm_name = f"FedAvg_NoPrune_SignFlip20_a1.0_s{s}"
        old_sf = HISTORICAL_50K_SF[s]
        new_full = cell00_results[arm_name]["final_clean_acc_full"]
        new_disj = cell00_results[arm_name]["final_clean_acc_disjoint"]
        delta = new_full - old_sf
        offset_deltas.append(delta)
        print(f"s{s:<5} | {old_sf:6.2f}%            | {new_full:6.2f}%               | {new_disj:6.2f}%               | {delta:+6.2f} pp")

    m_off, sd_off, se_off, t_off, p_off, ci_off = calc_paired_stats(offset_deltas)
    print("-" * 90)
    print(f"Mean Partition-Offset Shift (49.5k - 50k): {m_off:+.2f} +/- {sd_off:.2f} pp (SE={se_off:.2f} pp, t={t_off:.3f}, p={p_off:.4f}, 95% CI=[{ci_off[0]:+.2f}, {ci_off[1]:+.2f}])")

    if stage8_results:
        d_sketch_full, d_sketch_disj = [], []
        d_prune_full, d_prune_disj = [], []
        for s in seeds:
            c00 = cell00_results[f"FedAvg_NoPrune_SignFlip20_a1.0_s{s}"]
            sk = stage8_results[f"EdgeShield_NoPrune_SignFlip20_a1.0_s{s}"]
            pr = stage8_results[f"FedAvg_PruneOnly_SignFlip20_a1.0_s{s}"]

            d_sketch_full.append(sk["final_clean_acc_full"] - c00["final_clean_acc_full"])
            d_sketch_disj.append(sk["final_clean_acc_disjoint"] - c00["final_clean_acc_disjoint"])
            d_prune_full.append(pr["final_clean_acc_full"] - c00["final_clean_acc_full"])
            d_prune_disj.append(pr["final_clean_acc_disjoint"] - c00["final_clean_acc_disjoint"])

        m_sk_f, sd_sk_f, se_sk_f, t_sk_f, p_sk_f, ci_sk_f = calc_paired_stats(d_sketch_full)
        m_sk_d, sd_sk_d, se_sk_d, t_sk_d, p_sk_d, ci_sk_d = calc_paired_stats(d_sketch_disj)
        m_pr_f, sd_pr_f, se_pr_f, t_pr_f, p_pr_f, ci_pr_f = calc_paired_stats(d_prune_full)
        m_pr_d, sd_pr_d, se_pr_d, t_pr_d, p_pr_d, ci_pr_d = calc_paired_stats(d_prune_disj)

        print("\n--- Harness-Matched Paired Recovery (vs. 49.5k Cell (0,0) Baselines) ---")
        print(f"Cell (1,0) Sketch-Only (10k Full):     D = {m_sk_f:+.2f} +/- {sd_sk_f:.2f}% (SE={se_sk_f:.2f}%, t={t_sk_f:.3f}, p={p_sk_f:.4f}, 95% CI=[{ci_sk_f[0]:+.2f}%, {ci_sk_f[1]:+.2f}%])")
        print(f"Cell (1,0) Sketch-Only (9.5k Disjoint): D = {m_sk_d:+.2f} +/- {sd_sk_d:.2f}% (SE={se_sk_d:.2f}%, t={t_sk_d:.3f}, p={p_sk_d:.4f}, 95% CI=[{ci_sk_d[0]:+.2f}%, {ci_sk_d[1]:+.2f}%])")
        print(f"Cell (0,1) Prune-Only  (10k Full):     D = {m_pr_f:+.2f} +/- {sd_pr_f:.2f}% (SE={se_pr_f:.2f}%, t={t_pr_f:.3f}, p={p_pr_f:.4f}, 95% CI=[{ci_pr_f[0]:+.2f}%, {ci_pr_f[1]:+.2f}%])")
        print(f"Cell (0,1) Prune-Only  (9.5k Disjoint): D = {m_pr_d:+.2f} +/- {sd_pr_d:.2f}% (SE={se_pr_d:.2f}%, t={t_pr_d:.3f}, p={p_pr_d:.4f}, 95% CI=[{ci_pr_d[0]:+.2f}%, {ci_pr_d[1]:+.2f}%])")
    print("=" * 90)


if __name__ == "__main__":
    main()