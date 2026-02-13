#!/usr/bin/env python3
"""
Dataset Statistics for GyroGem Chat QA Dataset
Run from: GyroGem/training/data/
Usage: python dataset_stats.py
"""

import json
import os
from collections import Counter, defaultdict

DATASET_PATH = "gyrogem_chat_qa_dataset.jsonl"


def load_dataset(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"  WARNING: JSON parse error on line {line_num}: {e}")
    return entries


def get_prefix(entry_id):
    """Extract prefix from ID like 'drill_corr_001' -> 'drill_corr'"""
    parts = entry_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return entry_id


def get_top_prefix(entry_id):
    """Extract top-level prefix like 'drill_corr_001' -> 'drill'"""
    return entry_id.split("_")[0]


def analyze_conversations(entry):
    """Analyze conversation structure of an entry."""
    convos = entry.get("conversations", [])
    roles = [c.get("role") for c in convos]
    has_system = "system" in roles
    num_turns = len(convos)
    assistant_msgs = [c["content"] for c in convos if c.get("role") == "assistant"]
    user_msgs = [c["content"] for c in convos if c.get("role") == "user"]
    total_assistant_chars = sum(len(m) for m in assistant_msgs)
    total_user_chars = sum(len(m) for m in user_msgs)
    return {
        "has_system": has_system,
        "num_turns": num_turns,
        "assistant_chars": total_assistant_chars,
        "user_chars": total_user_chars,
        "total_chars": total_assistant_chars + total_user_chars,
    }


def check_quality(entry):
    """Check for common quality issues."""
    issues = []
    entry_id = entry.get("id", "NO_ID")
    convos = entry.get("conversations", [])

    for c in convos:
        content = c.get("content", "")
        role = c.get("role", "")

        # Check for em dashes
        if "\u2014" in content:
            issues.append(f"em dash found in {role} message")

        # Check for double spaces
        if "  " in content:
            issues.append(f"double space found in {role} message")

        # Check assistant messages for THM reference
        if role == "assistant":
            if "The Human Mark" not in content and "THM" not in content:
                issues.append("assistant response does not reference THM")

    # Check for missing fields
    if "id" not in entry:
        issues.append("missing 'id' field")
    if "conversations" not in entry:
        issues.append("missing 'conversations' field")
    if "source_doc" not in entry:
        issues.append("missing 'source_doc' field")
    if "tags" not in entry:
        issues.append("missing 'tags' field")

    return issues


def print_section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found.")
        print(f"Run this script from the directory containing the dataset.")
        return

    print(f"Loading {DATASET_PATH}...")
    entries = load_dataset(DATASET_PATH)
    print(f"Loaded {len(entries)} entries.")

    # =============================================
    # 1. BASIC COUNTS
    # =============================================
    print_section("BASIC COUNTS")
    print(f"  Total entries: {len(entries)}")

    ids = [e.get("id", "NO_ID") for e in entries]
    unique_ids = set(ids)
    print(f"  Unique IDs: {len(unique_ids)}")
    if len(ids) != len(unique_ids):
        dupes = [i for i, count in Counter(ids).items() if count > 1]
        print(f"  DUPLICATE IDs: {dupes}")

    # =============================================
    # 2. ID PREFIX BREAKDOWN
    # =============================================
    print_section("ID PREFIX BREAKDOWN (detailed)")
    prefix_counts = Counter(get_prefix(e.get("id", "NO_ID")) for e in entries)
    for prefix, count in sorted(prefix_counts.items()):
        print(f"  {prefix:40s} {count:4d}")

    print_section("ID PREFIX BREAKDOWN (top-level)")
    top_prefix_counts = Counter(get_top_prefix(e.get("id", "NO_ID")) for e in entries)
    for prefix, count in sorted(top_prefix_counts.items()):
        print(f"  {prefix:40s} {count:4d}")

    # =============================================
    # 3. SOURCE DOCUMENT DISTRIBUTION
    # =============================================
    print_section("SOURCE DOCUMENT DISTRIBUTION")
    source_counts = Counter(e.get("source_doc", "NONE") for e in entries)
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(entries)
        print(f"  {source:40s} {count:4d}  ({pct:5.1f}%)")

    # =============================================
    # 4. TAG ANALYSIS
    # =============================================
    print_section("TAG FREQUENCY (top 40)")
    all_tags = []
    for e in entries:
        all_tags.extend(e.get("tags", []))
    tag_counts = Counter(all_tags)
    for tag, count in tag_counts.most_common(40):
        print(f"  {tag:40s} {count:4d}")

    print_section("TAG COVERAGE")
    print(f"  Total unique tags: {len(tag_counts)}")
    entries_with_tags = sum(1 for e in entries if e.get("tags"))
    entries_without_tags = len(entries) - entries_with_tags
    print(f"  Entries with tags: {entries_with_tags}")
    print(f"  Entries without tags: {entries_without_tags}")

    # =============================================
    # 5. CONVERSATION STRUCTURE
    # =============================================
    print_section("CONVERSATION STRUCTURE")
    analyses = [analyze_conversations(e) for e in entries]

    system_count = sum(1 for a in analyses if a["has_system"])
    no_system_count = len(analyses) - system_count
    print(f"  With system prompt: {system_count} ({100.0*system_count/len(entries):.1f}%)")
    print(f"  Without system prompt: {no_system_count} ({100.0*no_system_count/len(entries):.1f}%)")

    turn_counts = Counter(a["num_turns"] for a in analyses)
    print(f"\n  Turn count distribution:")
    for turns, count in sorted(turn_counts.items()):
        print(f"    {turns} turns: {count}")

    # =============================================
    # 6. RESPONSE LENGTH STATISTICS
    # =============================================
    print_section("RESPONSE LENGTH STATISTICS (characters)")
    assistant_lengths = [a["assistant_chars"] for a in analyses]
    user_lengths = [a["user_chars"] for a in analyses]
    total_lengths = [a["total_chars"] for a in analyses]

    def stats(values, label):
        values_sorted = sorted(values)
        n = len(values_sorted)
        mean_val = sum(values_sorted) / n
        median_val = values_sorted[n // 2]
        p10 = values_sorted[int(n * 0.1)]
        p90 = values_sorted[int(n * 0.9)]
        print(f"  {label}:")
        print(f"    Min: {values_sorted[0]}, Max: {values_sorted[-1]}")
        print(f"    Mean: {mean_val:.0f}, Median: {median_val}")
        print(f"    P10: {p10}, P90: {p90}")
        print(f"    Total: {sum(values_sorted):,}")

    stats(assistant_lengths, "Assistant response length")
    stats(user_lengths, "User prompt length")
    stats(total_lengths, "Total conversation length")

    # =============================================
    # 7. DISPLACEMENT RISK COVERAGE
    # =============================================
    print_section("DISPLACEMENT RISK COVERAGE")
    risk_tags = {"gtd", "ivd", "iad", "iid"}
    risk_coverage = defaultdict(list)
    for e in entries:
        tags = set(e.get("tags", []))
        for risk in risk_tags:
            if risk in tags:
                risk_coverage[risk].append(e.get("id"))

    for risk in ["gtd", "ivd", "iad", "iid"]:
        count = len(risk_coverage[risk])
        print(f"  {risk.upper():6s} tagged entries: {count}")

    # Entries with no risk tag
    no_risk = [e.get("id") for e in entries if not risk_tags.intersection(set(e.get("tags", [])))]
    print(f"  No risk tag: {len(no_risk)}")

    # =============================================
    # 8. CONTENT CATEGORY ESTIMATE
    # =============================================
    print_section("CONTENT CATEGORY ESTIMATE (by ID prefix)")
    categories = {
        "Core framework": ["thm_md"],
        "Briefing": ["thm_brief"],
        "Terminology": ["thm_terms"],
        "Specifications": ["thm_specs"],
        "Grammar": ["thm_grammar"],
        "Paper": ["thm_paper"],
        "Positive path / Applied": ["thm_pos", "thm_app"],
        "Role handling": ["thm_role"],
        "Structural / Theoretical": ["thm_struct"],
        "Drills": ["drill"],
        "Clarifications": ["clar"],
    }

    for cat_name, prefixes in categories.items():
        count = sum(
            1 for e in entries
            if any(get_prefix(e.get("id", "")).startswith(p) for p in prefixes)
        )
        pct = 100.0 * count / len(entries)
        print(f"  {cat_name:35s} {count:4d}  ({pct:5.1f}%)")

    # =============================================
    # 9. QUALITY CHECKS
    # =============================================
    print_section("QUALITY CHECKS")
    issues_by_entry = {}
    for e in entries:
        issues = check_quality(e)
        if issues:
            issues_by_entry[e.get("id", "NO_ID")] = issues

    if issues_by_entry:
        print(f"  Entries with issues: {len(issues_by_entry)}")
        # Group by issue type
        issue_type_counts = Counter()
        for entry_id, issues in issues_by_entry.items():
            for issue in issues:
                issue_type_counts[issue] += 1

        print(f"\n  Issue type summary:")
        for issue, count in issue_type_counts.most_common():
            print(f"    {issue}: {count}")

        print(f"\n  First 10 entries with issues:")
        for i, (entry_id, issues) in enumerate(issues_by_entry.items()):
            if i >= 10:
                print(f"    ... and {len(issues_by_entry) - 10} more")
                break
            print(f"    {entry_id}: {'; '.join(issues)}")
    else:
        print("  No issues found.")

    # =============================================
    # 10. ANCHOR ESTIMATE
    # =============================================
    print_section("ANCHOR DENSITY ESTIMATE")
    anchor_keywords = [
        "Governance Traceability Displacement",
        "Information Variety Displacement",
        "Inference Accountability Displacement",
        "Intelligence Integrity Displacement",
        "Common Source Consensus",
        "Direct Authority",
        "Indirect Authority",
        "Direct Agency",
        "Indirect Agency",
    ]

    anchor_count = 0
    for e in entries:
        text = json.dumps(e.get("conversations", []))
        if any(kw in text for kw in anchor_keywords):
            anchor_count += 1

    print(f"  Entries containing anchor keywords: {anchor_count} ({100.0*anchor_count/len(entries):.1f}%)")
    print(f"  Entries without anchor keywords: {len(entries) - anchor_count}")
    print(f"  NOTE: If oversampling anchors at high multiplier,")
    print(f"  effective training set will be heavily skewed.")

    # =============================================
    # SUMMARY
    # =============================================
    print_section("SUMMARY")
    print(f"  Total entries: {len(entries)}")
    print(f"  Unique IDs: {len(unique_ids)}")
    print(f"  Duplicate IDs: {len(ids) - len(unique_ids)}")
    print(f"  Source documents: {len(source_counts)}")
    print(f"  Unique tags: {len(tag_counts)}")
    print(f"  System prompts: {system_count} ({100.0*system_count/len(entries):.1f}%)")
    print(f"  Entries with quality issues: {len(issues_by_entry)}")
    print(f"  Anchor-qualifying entries: {anchor_count} ({100.0*anchor_count/len(entries):.1f}%)")
    total_chars = sum(a["total_chars"] for a in analyses)
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens (chars/4): ~{total_chars//4:,}")


if __name__ == "__main__":
    main()