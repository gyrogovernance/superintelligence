#!/usr/bin/env python3
"""
Search for potentially swapped definitions.
"""

import json
import re

DATASET_PATH = "gyrogem_chat_qa_dataset.jsonl"

# Patterns that would indicate an ERROR (wrong definition)
ERROR_PATTERNS = [
    # Indirect Agency described as human
    (r"Indirect Agency[^.]{0,30}human subject", "Indirect Agency called human subject"),
    (r"Indirect Agency[^.]{0,30}human capacity", "Indirect Agency called human capacity"),
    (r"Indirect Agency \(human", "Indirect Agency parenthetical says human"),
    (r"Indirect Agency.*?human.*?receiving", "Indirect Agency associated with human receiving"),

    # Direct Agency described as artificial
    (r"Direct Agency[^.]{0,30}artificial subject", "Direct Agency called artificial subject"),
    (r"Direct Agency[^.]{0,30}artificial capacity", "Direct Agency called artificial capacity"),
    (r"Direct Agency \(artificial", "Direct Agency parenthetical says artificial"),
    (r"Direct Agency.*?artificial.*?processing", "Direct Agency associated with artificial processing"),

    # IVD called Inference (wrong first word)
    (r"IVD[^.]{0,10}Inference Variety", "IVD expanded as Inference Variety"),
    (r"Information Variety Displacement \(IAD", "IVD name with IAD acronym"),

    # IAD called Information (wrong first word)
    (r"IAD[^.]{0,10}Information", "IAD expanded starting with Information"),
    (r"Inference Accountability Displacement \(IVD", "IAD name with IVD acronym"),

    # IID called Information (wrong first word)
    (r"IID[^.]{0,10}Information Integrity", "IID expanded as Information Integrity"),
    (r"Intelligence Integrity Displacement \(IVD", "IID name with IVD acronym"),

    # Wrong risk numbers
    (r"IAD[^.]{0,30}second (displacement )?risk", "IAD called second risk (should be third)"),
    (r"IAD[^.]{0,30}first (displacement )?risk", "IAD called first risk (should be third)"),
    (r"IAD[^.]{0,30}fourth (displacement )?risk", "IAD called fourth risk (should be third)"),
    (r"IVD[^.]{0,30}third (displacement )?risk", "IVD called third risk (should be second)"),
    (r"IVD[^.]{0,30}first (displacement )?risk", "IVD called first risk (should be second)"),
    (r"IVD[^.]{0,30}fourth (displacement )?risk", "IVD called fourth risk (should be second)"),
    (r"IID[^.]{0,30}second (displacement )?risk", "IID called second risk (should be fourth)"),
    (r"IID[^.]{0,30}third (displacement )?risk", "IID called third risk (should be fourth)"),
    (r"IID[^.]{0,30}first (displacement )?risk", "IID called first risk (should be fourth)"),
    (r"GTD[^.]{0,30}second (displacement )?risk", "GTD called second risk (should be first)"),
    (r"GTD[^.]{0,30}third (displacement )?risk", "GTD called third risk (should be first)"),
    (r"GTD[^.]{0,30}fourth (displacement )?risk", "GTD called fourth risk (should be first)"),
]


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Scanning {len(entries)} entries for potential errors...\n")

    found_any = False

    for entry in entries:
        entry_id = entry.get("id", "NO_ID")
        convos = entry.get("conversations", [])

        for conv in convos:
            if conv.get("role") != "assistant":
                continue

            text = conv.get("content", "")

            for pattern, description in ERROR_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    found_any = True
                    print(f"{'='*60}")
                    print(f"POTENTIAL ERROR: {description}")
                    print(f"Entry ID: {entry_id}")
                    print(f"Match: {matches}")
                    print("\nContext (first 500 chars):")
                    print(f"{text[:500]}...")
                    print()

    if not found_any:
        print("No potential errors found.")


if __name__ == "__main__":
    main()
