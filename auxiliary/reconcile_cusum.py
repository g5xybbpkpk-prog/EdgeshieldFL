import json
import statistics
import math

with open("bridge2_results.json") as f:
    arm = json.load(f)["results"]["EdgeShield_EMA_SignFlip20"]

rho_c = arm["rho_c"]
cusum = arm["cusum_states"]

# 1. First crossing rounds for attackers (16, 17, 18, 19)
print("=== ATTACKER FIRST CROSSING ROUNDS (h = 5.0) ===")
first_crossings = {}
for i in range(16, 20):
    crossed = None
    for r in range(5, len(cusum)):
        if cusum[r][i] >= 5.0:
            crossed = r
            break
    first_crossings[i] = crossed
    total_floored = sum(cusum[r][i] >= 5.0 for r in range(len(cusum)))
    print(f"Attacker {i}: First crossing = Round {crossed} | Total rounds floored = {total_floored}/101")

# Also check honest clients first crossing and flooring
print("\n=== HONEST CLIENTS CROSSING SUMMARY ===")
hon_crossings = {}
for i in range(16):
    crossed = None
    for r in range(5, len(cusum)):
        if cusum[r][i] >= 5.0:
            crossed = r
            break
    total_floored = sum(cusum[r][i] >= 5.0 for r in range(len(cusum)))
    if crossed is not None or total_floored > 0:
        print(f"Honest Client {i}: First crossing = Round {crossed} | Total rounds floored = {total_floored}/101")

# 2. Phase-resolved drift table
phases = {
    "Early (Rounds 5-30)": range(5, 31),
    "Mid (Rounds 31-60)": range(31, 61),
    "Late (Rounds 61-100)": range(61, 101)
}

print("\n" + "=" * 90)
print("PHASE-RESOLVED DRIFT TABLE (C=5.0, alpha=0.3, Seed 42, k=20 EMA)")
print("=" * 90)
print(f"{'Phase':<22} | {'Group':<10} | {'Mean rho_c':<12} | {'Std rho_c':<12} | {'Raw Drift':<14} | {'Std Drift (/0.1474)':<18}")
print("-" * 90)

for pname, prange in phases.items():
    hon_rhos = []
    att_rhos = []
    for r in prange:
        hon_rhos.extend([rho_c[r][i] for i in range(16)])
        att_rhos.extend([rho_c[r][i] for i in range(16, 20)])
    
    m_h, s_h = statistics.mean(hon_rhos), statistics.stdev(hon_rhos)
    m_a, s_a = statistics.mean(att_rhos), statistics.stdev(att_rhos)
    
    raw_drift_h = m_h - 0.01
    raw_drift_a = m_a - 0.01
    
    std_drift_h = raw_drift_h / 0.1474
    std_drift_a = raw_drift_a / 0.1474
    
    print(f"{pname:<22} | {'Honest':<10} | {m_h:+.4f}       | {s_h:.4f}       | {raw_drift_h:+.4f}         | {std_drift_h:+.4f}")
    print(f"{'':<22} | {'Attacker':<10} | {m_a:+.4f}       | {s_a:.4f}       | {raw_drift_a:+.4f}         | {std_drift_a:+.4f}")
    print("-" * 90)

# 3. Individual attacker drift and cumulative progression during Early Phase
print("\n=== PER-ATTACKER EARLY-PHASE DRIFT & FIRST CROSSING ANALYSIS ===")
for i in range(16, 20):
    att_early_rhos = [rho_c[r][i] for r in range(5, 31)]
    m = statistics.mean(att_early_rhos)
    std = statistics.stdev(att_early_rhos)
    drift = (m - 0.01) / 0.1474
    print(f"Attacker {i}: Early mean rho_c = {m:+.4f} (std = {std:.4f}) | Mean drift/round = {drift:+.4f} | First cross = Round {first_crossings[i]}")

# 4. Check empirical std of clean / honest clients vs frozen 0.1474
all_hon_rhos = []
for r in range(5, 101):
    all_hon_rhos.extend([rho_c[r][i] for i in range(16)])
overall_s_h = statistics.stdev(all_hon_rhos)
overall_m_h = statistics.mean(all_hon_rhos)
print(f"\nEmpirical Honest rho_c over all rounds 5-100: Mean = {overall_m_h:+.4f}, Std = {overall_s_h:.4f} (vs frozen 0.1474)")
