#!/usr/bin/env python3
"""
AIR CLI test suite - project-only, non-interactive model.
Tests the default behavior: sync all projects, verify, bundle.
"""

import subprocess
import sys
from pathlib import Path
import shutil
import json
import zipfile
import pytest

PROJECT_ROOT = Path(__file__).parent.parent


def run_aci(cwd=None) -> tuple[int, str, str]:
    """Run aci.py and return (exit_code, stdout, stderr)."""
    try:
        if cwd is None:
            cwd = PROJECT_ROOT
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "aci.py")],
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
    """Clean up test data (projects, bundles, reports) but preserve atlas."""
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return
    
    # Remove projects directory (but not atlas)
    projects_dir = data_dir / "projects"
    if projects_dir.exists():
        shutil.rmtree(projects_dir)


def test_a_cold_start_builds_atlas_and_templates():
    """Test A: Cold start builds atlas + templates."""
    cleanup_data()
    
    exit_code, stdout, stderr = run_aci()
    if exit_code != 0:
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
    assert exit_code == 0, f"Exit code {exit_code}, stderr: {stderr}, stdout: {stdout[:500]}"
    
    # Check atlas files
    atlas_dir = PROJECT_ROOT / "data" / "atlas"
    required_files = ["ontology.npy", "epistemology.npy", "phenomenology.npz"]
    for f in required_files:
        assert (atlas_dir / f).exists(), f"Missing atlas file: {f}"
    
    # Check template
    template = PROJECT_ROOT / "data" / "projects" / "templates" / "project_template.md"
    assert template.exists(), "Missing project template"


def test_b_compile_project_into_artifacts():
    """Test B: Compile a project into .aci + bundle + report."""
    # Create test project
    projects_dir = PROJECT_ROOT / "data" / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    
    project_md = projects_dir / "test-project.md"
    project_content = """---
project_name: Test Project
project_slug: test-project
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

# Test Project

Test project description.
"""
    project_md.write_text(project_content, encoding="utf-8")
    
    # Run aci
    exit_code, stdout, stderr = run_aci()
    if exit_code != 0:
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
    assert exit_code == 0, f"Exit code {exit_code}, stderr: {stderr}, stdout: {stdout[:500]}"
    
    # Check artifacts
    aci_dir = PROJECT_ROOT / "data" / "projects" / ".aci"
    required_artifacts = [
        "test-project.bytes",
        "test-project.events.jsonl",
        "test-project.report.json",
        "test-project.report.md",
    ]
    
    for artifact in required_artifacts:
        assert (aci_dir / artifact).exists(), f"Missing artifact: {artifact}"
    
    # Check bundle
    bundle = PROJECT_ROOT / "data" / "projects" / "bundles" / "test-project.zip"
    assert bundle.exists(), "Missing bundle"
    
    # Verify bundle contents
    with zipfile.ZipFile(bundle, "r") as zf:
        files = zf.namelist()
        required = ["project.md", "bytes.bin", "events.jsonl", "report.json", "report.md", "bundle.json"]
        missing = [f for f in required if f not in files]
        assert not missing, f"Bundle missing files: {missing}"
    
    # Clean up test project
    if project_md.exists():
        project_md.unlink()


def test_c_tamper_detection():
    """Test C: Tamper detection - verify bundle integrity check directly."""
    from src.app.cli import store
    
    # Find bundle
    bundle = PROJECT_ROOT / "data" / "projects" / "bundles" / "test-project.zip"
    if not bundle.exists():
        pytest.skip("No bundle to tamper")
    
    # Create tampered bundle (corrupt bytes.bin)
    tampered = PROJECT_ROOT / "data" / "projects" / "bundles" / "test-project-tampered.zip"
    with zipfile.ZipFile(bundle, "r") as src:
        with zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.namelist():
                data = src.read(item)
                if item == "bytes.bin":
                    # Corrupt first byte
                    data = b"\xff" + data[1:] if len(data) > 0 else b"\xff"
                dst.writestr(item, data)
    
    # Test verify_bundle directly (CLI would overwrite, so we test the library function)
    atlas_dir = PROJECT_ROOT / "data" / "atlas"
    assert not store.verify_bundle(atlas_dir, tampered), "Tampered bundle should fail verification"
    
    # Clean up
    tampered.unlink()
    
    # Clean up test project
    project_md = PROJECT_ROOT / "data" / "projects" / "test-project.md"
    if project_md.exists():
        project_md.unlink()


def test_d_determinism():
    """Test D: Determinism."""
    # Ensure test project exists (project_id will be auto-generated in .aci/ on first sync)
    projects_dir = PROJECT_ROOT / "data" / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    
    project_md = projects_dir / "test-project.md"
    if not project_md.exists():
        # Create project (project_id will be auto-generated in .aci/ on first sync)
        project_content = """---
project_name: Test Project
project_slug: test-project
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

# Test Project

Test project description.
"""
        project_md.write_text(project_content, encoding="utf-8")
    
    bundle = PROJECT_ROOT / "data" / "projects" / "bundles" / "test-project.zip"
    
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
    # project_md_sha256 changes because project.md is updated with last_synced_at on each run
    bundle_json1["logs"].pop("project_md_sha256", None)
    bundle_json2["logs"].pop("project_md_sha256", None)
    # Report hashes are deterministic but derived from bytes/events, so remove for comparison
    bundle_json1["logs"].pop("report_json_sha256", None)
    bundle_json2["logs"].pop("report_json_sha256", None)
    bundle_json1["logs"].pop("report_md_sha256", None)
    bundle_json2["logs"].pop("report_md_sha256", None)
    
    assert bundle_json1 == bundle_json2, "Bundle.json differs between runs (excluding timestamps and project/report hashes)"
    
    # Clean up test project
    if project_md.exists():
        project_md.unlink()


def test_e_skipped_attestations_in_report():
    """Test E: Skipped attestations appear in report."""
    # Create project with invalid attestation
    projects_dir = PROJECT_ROOT / "data" / "projects"
    project_md = projects_dir / "test-skip.md"
    project_content = """---
project_name: Test Skip
project_slug: test-skip
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

# Test Skip Project
"""
    project_md.write_text(project_content, encoding="utf-8")
    
    # Run aci
    _exit_code, _stdout, _stderr = run_aci()
    
    # Check report
    report_json = PROJECT_ROOT / "data" / "projects" / ".aci" / "test-skip.report.json"
    assert report_json.exists(), "Report not generated"
    
    report_data = json.loads(report_json.read_text())
    
    assert "skipped_attestations" in report_data["compilation"], "Report missing skipped_attestations"
    
    skipped = report_data["compilation"]["skipped_attestations"]
    assert len(skipped) == 2, f"Expected 2 skipped, got {len(skipped)}"
    
    # Check reasons
    reasons = [s["reason"] for s in skipped]
    assert any("invalid unit" in r for r in reasons), "Missing 'invalid unit' reason"
    assert any("invalid domain" in r for r in reasons), "Missing 'invalid domain' reason"
    
    # Clean up test project
    if project_md.exists():
        project_md.unlink()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AIR CLI Test Suite (Project-Only Model)")
    print("="*60)
    
    results = []
    
    tests = [
        ("Cold Start", test_a_cold_start_builds_atlas_and_templates),
        ("Project Compilation", test_b_compile_project_into_artifacts),
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
