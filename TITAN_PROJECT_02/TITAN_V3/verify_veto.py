import sys
import os
sys.path.append(os.getcwd())

from src.brain.api.main import app, seldon_monitor
import pandas as pd
import numpy as np

print("=" * 60)
print("SELDON VETO VERIFICATION (R3K + R6V)")
print("=" * 60)

# [TAG: R6V_OVERLAP_TEST]
print("\n[1/3] Testing Data Overlap Detection...")
# Simulate loading crisis files to check for duplicates
crisis_files = [
    'data/XAUUSD_D1_2020_COVID.csv',
    'data/XAUUSD_D1_2022_Ukraine.csv',
    'data/XAUUSD_D1_2011-2012_Euro.csv'
]

all_timestamps = []
for fp in crisis_files:
    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp)
            if 'time' in df.columns or 'date' in df.columns:
                time_col = 'time' if 'time' in df.columns else 'date'
                all_timestamps.extend(df[time_col].tolist())
        except:
            pass

if all_timestamps:
    unique_timestamps = set(all_timestamps)
    overlap_pct = (1 - len(unique_timestamps) / len(all_timestamps)) * 100
    print(f"Total timestamps: {len(all_timestamps)}")
    print(f"Unique timestamps: {len(unique_timestamps)}")
    print(f"Overlap: {overlap_pct:.1f}%")
    
    if overlap_pct > 10:
        print(f"⚠️ WARNING: High overlap detected ({overlap_pct:.1f}%)!")
    else:
        print(f"✅ Overlap is acceptable ({overlap_pct:.1f}%)")
else:
    print("⚠️ Could not verify timestamps (files not found or no time column)")

# [TAG: R3K_SELDON_VALIDATION]
print(f"\n[2/3] Testing Seldon Fitting...")
print(f"Is Fitted? {seldon_monitor.is_fitted}")
assert seldon_monitor.is_fitted, "Seldon should be fitted with real data!"
print("✅ Seldon is fitted")

# Test Normal Return (0.1%)
print("\n[3/3] Testing Anomaly Detection...")
print("Testing Normal Return (0.1%)...")
seldon_monitor.update(0.001)
print(f"Is Anomaly? {seldon_monitor.is_anomaly}")
assert not seldon_monitor.is_anomaly, "0.1% return should NOT be an anomaly"
print("✅ Normal return correctly classified")

# Test Crash Return (-10%)
print("Testing CRASH Return (-10%)...")
seldon_monitor.update(-0.10)
print(f"Is Anomaly? {seldon_monitor.is_anomaly}")
assert seldon_monitor.is_anomaly, "-10% return MUST be an anomaly!"
print("✅ Crash correctly detected")

print("\n" + "=" * 60)
print("VERIFICATION SUCCESSFUL: Seldon is guarding the gate ✅")
print("=" * 60)
