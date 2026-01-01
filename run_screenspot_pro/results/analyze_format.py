#!/usr/bin/env python3
"""
Analyze raw_response formatting in JSON files.
Check how many responses match the expected (x, y) coordinate format.
"""

import json
import re
from pathlib import Path

# Files to analyze
JSON_FILES = [
    "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/gta1_baseline_100.json",
    "/Users/vincent/Desktop/test/GTA1/run_screenspot_pro/results/gta1_distance_reward_100.json",
]

# Expected format: (number, number) - with optional spaces and decimal points
COORD_PATTERN = re.compile(r"^\s*\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*$")


def analyze_json(json_path: str) -> dict:
    """
    Analyze raw_response formatting in a JSON file.
    Returns stats about formatting.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    details = data.get("details", [])
    total = len(details)
    
    correct_format = 0
    wrong_format = 0
    wrong_format_examples = []
    
    for entry in details:
        raw_response = entry.get("raw_response", "")
        
        if COORD_PATTERN.match(raw_response):
            correct_format += 1
        else:
            wrong_format += 1
            if len(wrong_format_examples) < 10:  # Keep first 10 examples
                wrong_format_examples.append({
                    "raw_response": raw_response[:200],  # Truncate long responses
                    "prompt": entry.get("prompt_to_evaluate", ""),
                })
    
    return {
        "total": total,
        "correct_format": correct_format,
        "wrong_format": wrong_format,
        "correct_pct": correct_format / total * 100 if total > 0 else 0,
        "wrong_pct": wrong_format / total * 100 if total > 0 else 0,
        "wrong_format_examples": wrong_format_examples,
    }


def main():
    for json_path in JSON_FILES:
        print(f"\n{'='*60}")
        print(f"File: {Path(json_path).name}")
        print(f"{'='*60}")
        
        if not Path(json_path).exists():
            print(f"  ERROR: File not found!")
            continue
        
        stats = analyze_json(json_path)
        
        print(f"\nTotal entries: {stats['total']}")
        print(f"Correct format (x, y): {stats['correct_format']} ({stats['correct_pct']:.1f}%)")
        print(f"Wrong format: {stats['wrong_format']} ({stats['wrong_pct']:.1f}%)")
        
        if stats['wrong_format_examples']:
            print(f"\nExamples of wrong format (up to 10):")
            for i, ex in enumerate(stats['wrong_format_examples'], 1):
                print(f"\n  [{i}] Prompt: {ex['prompt']}")
                print(f"      Response: {ex['raw_response']}")


if __name__ == "__main__":
    main()

