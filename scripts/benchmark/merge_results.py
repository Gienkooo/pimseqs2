import json
import sys
import os

# Usage: python3 merge_results.py <hyperfine.json> <energy.txt>

if len(sys.argv) < 3:
    sys.exit(1)

hf_file = sys.argv[1]
en_file = sys.argv[2]

if not os.path.exists(en_file):
    print(f"Warning: {en_file} not found. Skipping merge.")
    sys.exit(0)

with open(hf_file, "r") as f:
    data = json.load(f)

with open(en_file, "r") as f:
    # Filter out empty lines
    energies = [float(line.strip()) for line in f if line.strip()]

energy_idx = 0

# Hyperfine executes sequentially; map energy values to results in order
for result in data.get("results", []):
    num_runs = len(result["times"])

    # Extract energy readings for this specific command configuration
    current_energies = energies[energy_idx : energy_idx + num_runs]

    result["energies"] = current_energies

    if current_energies:
        result["energy_mean"] = sum(current_energies) / len(current_energies)
        result["energy_min"] = min(current_energies)
        result["energy_max"] = max(current_energies)

        # Calculate Average Power (Watts) = Energy (J) / Time (s)
        if result["mean"] > 0:
            result["power_avg_watts"] = result["energy_mean"] / result["mean"]

    energy_idx += num_runs

print(json.dumps(data, indent=2))
