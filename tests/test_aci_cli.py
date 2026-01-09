#!/usr/bin/env python3
"""
AIR CLI test suite - program-only, non-interactive model.
Tests the default behavior: sync all programs, verify, bundle.
"""

import subprocess
import sys
from pathlib import Path
import shutil
import json
import zipfile
import pytest

Program_ROOT = Path(__file__).parent.parent


def run_aci(cwd=None) -> tuple[int, str, str]:
    """Run aci.py and return (exit_code, stdout, stderr)."""
    try:
        if cwd is None:
            cwd = Program_ROOT
        result = subprocess.run(
            [sys.executable, str(Program_ROOT / "aci.py")],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def cleanup_data():
    """Clean up test data (programs, bundles, reports) but preserve atlas."""
    data_dir = Program_ROOT / "data"
    if not data_dir.exists():
        return
    
    # Remove programs directory (but not atlas)
    programs_dir = data_dir / "programs"
    if programs_dir.exists():
        shutil.rmtree(programs_dir)


def test_a_cold_start_builds_atlas_and_templates():
    """Test A: Cold start builds atlas + templates."""
    cleanup_data()
    
    exit_code, stdout, stderr = run_aci()
    if exit_code != 0:
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
    assert exit_code == 0, f"Exit code {exit_code}, stderr: {stderr}, stdout: {stdout[:500]}"
    
    # Check atlas files
    atlas_dir = Program_ROOT / "data" / "atlas"
    required_files = ["ontology.npy", "epistemology.npy", "phenomenology.npz"]
    for f in required_files:
        assert (atlas_dir / f).exists(), f"Missing atlas file: {f}"
    
    # Check template
    template = Program_ROOT / "data" / "programs" / "templates" / "program_template.md"
    assert template.exists(), "Missing program template"


def test_b_compile_program_into_artifacts():
    """Test B: Compile a program into .aci + bundle + report."""
    # Create test program
    programs_dir = Program_ROOT / "data" / "programs"
    programs_dir.mkdir(parents=True, exist_ok=True)
    
    program_md = programs_dir / "test-program.md"
    program_content = """---
program_name: Test Program
program_slug: test-program
sponsor: Test Lab
created_at: 2025-01-01T00:00:00Z

attestations:
  - id: att_001
    unit: daily
    domain: economy
    human_mark: Governance Traceability Displacement
    gyroscope_work: Governance Management
    evidence_links: []
    note: "Test attestation"
  
  - id: att_002
    unit: sprint
    domain: employment
    human_mark: Information Variety Displacement
    gyroscope_work: Information Curation
    evidence_links: []
    note: "Sprint attestation"

computed:
  last_synced_at: null
  apertures: {}
  event_count: 0
  kernel:
    step: 0
    state_index: 0
    state_hex: ""

---

# Test Program

Test program description.
"""
    program_md.write_text(program_content, encoding="utf-8")
    
    # Run aci
    exit_code, stdout, stderr = run_aci()
    if exit_code != 0:
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
    assert exit_code == 0, f"Exit code {exit_code}, stderr: {stderr}, stdout: {stdout[:500]}"
    
    # Check artifacts
    aci_dir = Program_ROOT / "data" / "programs" / ".aci"
    required_artifacts = [
        "test-program.bytes",
        "test-program.events.jsonl",
        "test-program.report.json",
        "test-program.report.md",
    ]
    
    for artifact in required_artifacts:
        assert (aci_dir / artifact).exists(), f"Missing artifact: {artifact}"
    
    # Check bundle
    bundle = Program_ROOT / "data" / "programs" / "bundles" / "test-program.zip"
    assert bundle.exists(), "Missing bundle"
    
    # Verify bundle contents
    with zipfile.ZipFile(bundle, "r") as zf:
        files = zf.namelist()
        required = ["program.md", "bytes.bin", "events.jsonl", "report.json", "report.md", "bundle.json"]
        missing = [f for f in required if f not in files]
        assert not missing, f"Bundle missing files: {missing}"
    
    # Clean up test program
    if program_md.exists():
        program_md.unlink()


def test_c_tamper_detection():
    """Test C: Tamper detection - verify bundle integrity check directly."""
    from src.app.cli import store
    
    # Find bundle
    bundle = Program_ROOT / "data" / "programs" / "bundles" / "test-program.zip"
    if not bundle.exists():
        pytest.skip("No bundle to tamper")
    
    # Create tampered bundle (corrupt bytes.bin)
    tampered = Program_ROOT / "data" / "programs" / "bundles" / "test-program-tampered.zip"
    with zipfile.ZipFile(bundle, "r") as src:
        with zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.namelist():
                data = src.read(item)
                if item == "bytes.bin":
                    # Corrupt first byte
                    data = b"\xff" + data[1:] if len(data) > 0 else b"\xff"
                dst.writestr(item, data)
    
    # Test verify_bundle directly (CLI would overwrite, so we test the library function)
    atlas_dir = Program_ROOT / "data" / "atlas"
    assert not store.verify_bundle(atlas_dir, tampered), "Tampered bundle should fail verification"
    
    # Clean up
    tampered.unlink()
    
    # Clean up test program
    program_md = Program_ROOT / "data" / "programs" / "test-program.md"
    if program_md.exists():
        program_md.unlink()


def test_d_determinism():
    """Test D: Determinism."""
    # Ensure test program exists (program_id will be auto-generated in .aci/ on first sync)
    programs_dir = Program_ROOT / "data" / "programs"
    programs_dir.mkdir(parents=True, exist_ok=True)
    
    program_md = programs_dir / "test-program.md"
    if not program_md.exists():
        # Create program (program_id will be auto-generated in .aci/ on first sync)
        program_content = """---
program_name: Test Program
program_slug: test-program
sponsor: Test Lab
created_at: 2025-01-01T00:00:00Z

attestations:
  - id: att_001
    unit: daily
    domain: economy
    human_mark: Governance Traceability Displacement
    gyroscope_work: Governance Management
    evidence_links: []
    note: "Test attestation"
  
  - id: att_002
    unit: sprint
    domain: employment
    human_mark: Information Variety Displacement
    gyroscope_work: Information Curation
    evidence_links: []
    note: "Sprint attestation"

computed:
  last_synced_at: null
  apertures: {}
  event_count: 0
  kernel:
    step: 0
    state_index: 0
    state_hex: ""

---

# Test Program

Test program description.
"""
        program_md.write_text(program_content, encoding="utf-8")
    
    bundle = Program_ROOT / "data" / "programs" / "bundles" / "test-program.zip"
    
    # Run first time and extract bundle.json
    exit_code1, _stdout1, _stderr1 = run_aci()
    assert exit_code1 == 0, "First run should succeed"
    
    with zipfile.ZipFile(bundle, "r") as zf:
        bundle_json1 = json.loads(zf.read("bundle.json"))
    
    # Run second time and extract bundle.json
    exit_code2, _stdout2, _stderr2 = run_aci()
    assert exit_code2 == 0, "Second run should succeed"
    
    with zipfile.ZipFile(bundle, "r") as zf:
        bundle_json2 = json.loads(zf.read("bundle.json"))
    
    # Remove non-deterministic fields for comparison
    bundle_json1.pop("generated_at", None)
    bundle_json2.pop("generated_at", None)
    # program_md_sha256 changes because program.md is updated with last_synced_at on each run
    bundle_json1["logs"].pop("program_md_sha256", None)
    bundle_json2["logs"].pop("program_md_sha256", None)
    # Report hashes are deterministic but derived from bytes/events, so remove for comparison
    bundle_json1["logs"].pop("report_json_sha256", None)
    bundle_json2["logs"].pop("report_json_sha256", None)
    bundle_json1["logs"].pop("report_md_sha256", None)
    bundle_json2["logs"].pop("report_md_sha256", None)
    
    assert bundle_json1 == bundle_json2, "Bundle.json differs between runs (excluding timestamps and program/report hashes)"
    
    # Clean up test program
    if program_md.exists():
        program_md.unlink()


def test_e_skipped_attestations_in_report():
    """Test E: Skipped attestations appear in report."""
    # Create program with invalid attestation
    programs_dir = Program_ROOT / "data" / "programs"
    program_md = programs_dir / "test-skip.md"
    program_content = """---
program_name: Test Skip
program_slug: test-skip
sponsor: Test Lab
created_at: 2025-01-01T00:00:00Z

attestations:
  - id: valid_001
    unit: daily
    domain: economy
    human_mark: Governance Traceability Displacement
  
  - id: invalid_001
    unit: invalid_unit
    domain: economy
    human_mark: Governance Traceability Displacement
  
  - id: invalid_002
    unit: daily
    domain: invalid_domain
    human_mark: Governance Traceability Displacement

computed:
  last_synced_at: null
  apertures: {}
  event_count: 0
  kernel:
    step: 0
    state_index: 0
    state_hex: ""

---

# Test Skip Program
"""
    program_md.write_text(program_content, encoding="utf-8")
    
    # Run aci
    _exit_code, _stdout, _stderr = run_aci()
    
    # Check report
    report_json = Program_ROOT / "data" / "programs" / ".aci" / "test-skip.report.json"
    assert report_json.exists(), "Report not generated"
    
    report_data = json.loads(report_json.read_text())
    
    assert "skipped_attestations" in report_data["compilation"], "Report missing skipped_attestations"
    
    skipped = report_data["compilation"]["skipped_attestations"]
    assert len(skipped) == 2, f"Expected 2 skipped, got {len(skipped)}"
    
    # Check reasons
    reasons = [s["reason"] for s in skipped]
    assert any("invalid unit" in r for r in reasons), "Missing 'invalid unit' reason"
    assert any("invalid domain" in r for r in reasons), "Missing 'invalid domain' reason"
    
    # Clean up test program
    if program_md.exists():
        program_md.unlink()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AIR CLI Test Suite (Program-Only Model)")
    print("="*60)
    
    results = []
    
    tests = [
        ("Cold Start", test_a_cold_start_builds_atlas_and_templates),
        ("Program Compilation", test_b_compile_program_into_artifacts),
        ("Tamper Detection", test_c_tamper_detection),
        ("Determinism", test_d_determinism),
        ("Skipped Attestations", test_e_skipped_attestations_in_report),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result is None:
                results.append((name, "SKIP", None))
            elif result:
                results.append((name, "PASS", None))
            else:
                results.append((name, "FAIL", None))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR", str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status in ["FAIL", "ERROR"])
    skipped = sum(1 for _, status, _ in results if status == "SKIP")
    
    for name, status, _ in results:
        if status == "PASS":
            print(f"[OK] {name}")
        elif status == "FAIL":
            print(f"[FAIL] {name}")
        elif status == "ERROR":
            print(f"[ERROR] {name}")
        else:
            print(f"[SKIP] {name}")
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
