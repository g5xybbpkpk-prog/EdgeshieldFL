import json
import statistics

# Load files
with open("bridge2_results.json") as f:
    b2 = json.load(f)["results"]
with open("stage6B_alphasweep_results.json") as f:
    b6 = json.load(f)["results"]
with open("stage6C_alpha1_seeds_results.json") as f:
    c6 = json.load(f)["results"]

arms_to_check = {
    "alpha=0.3 (s42)": b2["EdgeShield_EMA_SignFlip20"],
    "alpha=0.1 (s42)": b6["EdgeShield_EMA_SignFlip20_a0.1"],
    "alpha=1.0 (s42)": b6["EdgeShield_EMA_SignFlip20_a1.0"],
    "alpha=1.0 (s43)": c6["EdgeShield_EMA_SignFlip20_a1.0_s43"],
    "alpha=1.0 (s44)": c6["EdgeShield_EMA_SignFlip20_a1.0_s44"],
}

phases = {
    "Early (5-30)": range(5, 31),
    "Mid (31-60)": range(31, 61),
    "Late (61-100)": range(61, 101)
}

print(f"{'Arm / Regime':<18} | {'Phase':<13} | {'Honest rho':<12} | {'Attacker rho':<14} | {'d_prime':<8} | {'Att Drift/rnd':<14}")
print("-" * 88)

for aname, arm in arms_to_check.items():
    rho_c = arm["rho_c"]
    for pname, prange in phases.items():
        h_vals = []
        a_vals = []
        for r in prange:
            h_vals.extend([rho_c[r][i] for i in range(16)])
            a_vals.extend([rho_c[r][i] for i in range(16, 20)])
        
        m_h, s_h = statistics.mean(h_vals), statistics.stdev(h_vals)
        m_a, s_a = statistics.mean(a_vals), statistics.stdev(a_vals)
        s_p = (0.5 * (s_h**2 + s_a**2))**0.5
        dp = (m_a - m_h) / s_p if s_p > 0 else 0
        att_drift = (m_a - 0.01) / 0.1474
        print(f"{aname:<18} | {pname:<13} | {m_h:+.4f}+-{s_h:.3f} | {m_a:+.4f}+-{s_a:.3f}   | {dp:.3f}    | {att_drift:+.4f}")
    print("-" * 88)
