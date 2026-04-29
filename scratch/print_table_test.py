import json

def print_table():
    aggregated = {
        "training_free": {"wer_random": 0.5, "wer_oracle": 0.1},
        "rover": {"wer_rover": 0.2},
        "methods": {
            "mlp_pool": {"test": {"selected_wer": 0.25}},
            "hierarchical_transformer": {"test": {"selected_wer": 0.15}}
        }
    }
    
    # gather all rows
    rows = []
    if aggregated.get("training_free"):
        for k, v in aggregated["training_free"].items():
            rows.append((k, v))
    if aggregated.get("rover"):
        for k, v in aggregated["rover"].items():
            rows.append((k, v))
    if aggregated.get("methods"):
        for name, data in aggregated["methods"].items():
            if "test" in data and "selected_wer" in data["test"]:
                rows.append((name, data["test"]["selected_wer"]))
    
    print("-" * 50)
    print(f"| {'Method':<30} | {'Test WER':<13} |")
    print("-" * 50)
    for name, wer in sorted(rows, key=lambda x: x[1]):
        print(f"| {name:<30} | {wer:>13.4f} |")
    print("-" * 50)

print_table()
