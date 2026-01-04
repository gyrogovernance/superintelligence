"""
Workspace I/O and replay functions.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import re
import json
import hashlib
from datetime import datetime

from src.app.cli import templates


def get_data_dir() -> Path:
    """Returns data/ directory in current working directory."""
    data_dir = Path.cwd() / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_projects_dir() -> Path:
    """Returns data/projects/ directory."""
    projects_dir = get_data_dir() / "projects"
    if not projects_dir.exists():
        projects_dir.mkdir(parents=True, exist_ok=True)
    return projects_dir


def get_aci_dir() -> Path:
    """Returns data/projects/.aci/ directory for compiled artifacts."""
    aci_dir = get_projects_dir() / ".aci"
    if not aci_dir.exists():
        aci_dir.mkdir(parents=True, exist_ok=True)
    return aci_dir


def get_bundles_dir() -> Path:
    """Returns data/projects/bundles/ directory."""
    bundles_dir = get_projects_dir() / "bundles"
    if not bundles_dir.exists():
        bundles_dir.mkdir(parents=True, exist_ok=True)
    return bundles_dir


def get_atlas_dir() -> Path:
    """Returns data/atlas/ directory."""
    atlas_dir = get_data_dir() / "atlas"
    return atlas_dir


def ensure_workspace() -> None:
    """Ensure all workspace directories exist. Called on CLI startup."""
    data = get_data_dir()
    # Create standard dirs
    (data / "atlas").mkdir(parents=True, exist_ok=True)
    get_projects_dir()
    get_aci_dir()
    get_bundles_dir()


def ensure_templates() -> None:
    """
    Ensure project template is available in projects directory.
    Template is named with underscore prefix so it's not processed as a project.
    Users copy this file to create new projects (e.g., copy _template.md to my-project.md).
    
    Always overwrites the template to keep it in sync with the code version.
    """
    projects_dir = get_projects_dir()
    project_template = projects_dir / "_template.md"
    
    # Always overwrite to keep template in sync with code
    project_template.write_text(templates.PROJECT_TEMPLATE_MD, encoding="utf-8")


def parse_bracket_value(text: str, pattern: str) -> int:
    """
    Parse a bracket value from text using pattern.
    Pattern should contain a capture group for the abbreviation.
    Example: pattern = r'GTD:\\s*\\[(\\d+)\\]' will match 'GTD: [5]' and return 5.
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            return 0
    return 0


def parse_notes_section(text: str) -> str:
    """Parse the NOTES section from markdown text."""
    # Match ## NOTES section (case-insensitive) and capture everything after until next ## or end of file
    notes_pattern = r'^##\s+NOTES\s*---?\s*\n(.*?)(?=^##|\Z)'
    match = re.search(notes_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if match:
        notes = match.group(1).strip()
        # Remove the placeholder text if it's the default
        if notes == "(Add context or key observations for this project)":
            return ""
        return notes
    return ""


def parse_project_from_markdown(project_md_path: Path) -> Tuple[str, Dict[str, int], Dict[str, int], str, str]:
    """
    Parse project from markdown body using bracket notation.
    Returns: (project_slug, domain_counts, principle_counts, unit, notes)
    
    domain_counts: {"economy": int, "employment": int, "education": int}
    principle_counts: {"GMT": int, "GTD": int, "ICV": int, "IVD": int, "IIA": int, "IAD": int, "ICI": int, "IID": int}
    unit: "daily" or "sprint" (defaults to "daily" if not specified)
    notes: free text from NOTES section (empty string if not present)
    
    Note: project_slug is derived from filename stem to avoid collisions.
    """
    text = project_md_path.read_text(encoding="utf-8")
    
    # Slug = filename stem (canonical, avoids collisions)
    project_slug = project_md_path.stem
    
    # Parse domain counts
    domain_counts = {
        "economy": parse_bracket_value(text, r'Economy[^:]*:\s*\[(\d+)\]'),
        "employment": parse_bracket_value(text, r'Employment[^:]*:\s*\[(\d+)\]'),
        "education": parse_bracket_value(text, r'Education[^:]*:\s*\[(\d+)\]'),
    }
    
    # Parse principle counts (alignment and displacement)
    principle_counts = {
        "GMT": parse_bracket_value(text, r'GMT\s+Alignment\s+Incidents:\s*\[(\d+)\]'),
        "GTD": parse_bracket_value(text, r'GTD\s+Displacement\s+Incidents:\s*\[(\d+)\]'),
        "ICV": parse_bracket_value(text, r'ICV\s+Alignment\s+Incidents:\s*\[(\d+)\]'),
        "IVD": parse_bracket_value(text, r'IVD\s+Displacement\s+Incidents:\s*\[(\d+)\]'),
        "IIA": parse_bracket_value(text, r'IIA\s+Alignment\s+Incidents:\s*\[(\d+)\]'),
        "IAD": parse_bracket_value(text, r'IAD\s+Displacement\s+Incidents:\s*\[(\d+)\]'),
        "ICI": parse_bracket_value(text, r'ICI\s+Alignment\s+Incidents:\s*\[(\d+)\]'),
        "IID": parse_bracket_value(text, r'IID\s+Displacement\s+Incidents:\s*\[(\d+)\]'),
    }
    
    # Parse unit (default to "daily" if not found)
    unit_match = re.search(r'Unit:\s*\[(daily|sprint)\]', text, re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(1).lower()
        if unit not in ["daily", "sprint"]:
            unit = "daily"
    else:
        unit = "daily"
    
    # Parse notes
    notes = parse_notes_section(text)
    
    return project_slug, domain_counts, principle_counts, unit, notes


def generate_attestations_from_counts(
    domain_counts: Dict[str, int],
    principle_counts: Dict[str, int],
    project_id: str,
    unit: str = "daily"
) -> list[Dict[str, Any]]:
    """
    Generate attestations from domain and principle counts.
    Following GGG_Methodology: all terms are used to sustain balance (no optional choices).
    
    Strategy:
    - Each displacement incident (GTD, IVD, IAD, IID) generates one attestation (goes to ledger)
    - Each alignment incident (GMT, ICV, IIA, ICI) is counted but doesn't generate ledger events
    - Distribute incidents across domains proportionally based on domain_counts
    - Use specified unit ("daily" or "sprint") for all attestations
    """
    attestations = []
    
    # THM displacement mappings (these go to ledger)
    thm_displacement_map = {
        "GTD": "governance traceability displacement",
        "IVD": "information variety displacement",
        "IAD": "inference accountability displacement",
        "IID": "intelligence integrity displacement",
    }
    
    # Gyroscope alignment mappings (for reporting only, not ledger)
    gyro_alignment_map = {
        "GMT": "governance management traceability",
        "ICV": "information curation variety",
        "IIA": "inference interaction accountability",
        "ICI": "intelligence cooperation integrity",
    }
    
    # Calculate total domain count for proportional distribution
    total_domain_count = sum(domain_counts.values())
    domains = ["economy", "employment", "education"]
    
    # Build proportional distribution weights (following GGG balance principle)
    # All domains with counts > 0 should be represented proportionally
    domain_weights = {}
    for domain in domains:
        domain_weights[domain] = domain_counts.get(domain, 0)
    
    # If no domain counts specified, distribute evenly (all terms sustain balance)
    if total_domain_count == 0:
        for domain in domains:
            domain_weights[domain] = 1  # Equal weight for all domains
        total_domain_count = 3
    
    # Helper function to select domain proportionally
    def select_domain_proportional(incident_idx: int, total_incidents: int) -> str:
        """
        Select domain proportionally based on domain_counts.
        Uses deterministic distribution following GGG balance principle.
        All terms are used to sustain balance - no arbitrary choices.
        """
        if total_incidents == 0:
            # Fallback to first domain with weight > 0
            for domain in domains:
                if domain_weights[domain] > 0:
                    return domain
            return domains[0]
        
        # Calculate cumulative weights for proportional distribution
        cumulative = 0
        thresholds = []
        for domain in domains:
            cumulative += domain_weights[domain]
            thresholds.append((cumulative, domain))
        
        # Deterministic proportional selection
        # Map incident index to position in total domain space
        position = (incident_idx * total_domain_count) // total_incidents
        position = position % total_domain_count
        
        # Find which domain this position falls into based on proportional weights
        for threshold, domain in thresholds:
            if position < threshold:
                return domain
        
        # Fallback to first domain with weight > 0
        for domain in domains:
            if domain_weights[domain] > 0:
                return domain
        return domains[0]
    
    # Generate attestations for displacement incidents (THM - goes to ledger)
    # Following GGG: all terms are used to sustain balance
    att_idx = 0
    total_displacement_incidents = sum(principle_counts.get(abbrev, 0) for abbrev in thm_displacement_map.keys())
    
    for abbrev, full_name in thm_displacement_map.items():
        count = principle_counts.get(abbrev, 0)
        for i in range(count):
            # Proportional distribution based on domain_counts (GGG balance)
            domain = select_domain_proportional(att_idx, total_displacement_incidents)
            
            attestations.append({
                "id": f"{abbrev.lower()}_{i+1}",
                "unit": unit,
                "domain": domain,
                "human_mark": full_name,
            })
            att_idx += 1
    
    # Generate attestations for alignment incidents (Gyroscope - reporting only)
    # Following GGG: all terms are used to sustain balance
    total_alignment_incidents = sum(principle_counts.get(abbrev, 0) for abbrev in gyro_alignment_map.keys())
    alignment_att_idx = 0
    
    for abbrev, full_name in gyro_alignment_map.items():
        count = principle_counts.get(abbrev, 0)
        for i in range(count):
            # Proportional distribution based on domain_counts (GGG balance)
            domain = select_domain_proportional(alignment_att_idx, total_alignment_incidents)
            
            attestations.append({
                "id": f"{abbrev.lower()}_{i+1}",
                "unit": unit,
                "domain": domain,
                "gyroscope_work": full_name,
            })
            alignment_att_idx += 1
    
    return attestations


def read_bytes(path: Path) -> bytes:
    """Read all bytes from binary log."""
    if not path.exists():
        return b""
    return path.read_bytes()


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read all events from JSONL file."""
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    for line in text.split("\n"):
        if line.strip():
            events.append(json.loads(line))
    return events


def file_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_from_logs(atlas_dir: Path, bytes_path: Path, events_path: Path):
    """Replay bytes + events from given file paths to reconstruct state."""
    from src.app.coordination import Coordinator
    from src.app.events import GovernanceEvent, Domain, EdgeID

    coord = Coordinator(atlas_dir)
    
    # Replay bytes
    for b in read_bytes(bytes_path):
        coord.step_byte(b)

    # Replay events
    for ev_dict in read_events(events_path):
        ev = GovernanceEvent(
            domain=Domain(ev_dict["domain"]),
            edge_id=EdgeID(ev_dict["edge_id"]),
            magnitude=ev_dict["magnitude"],
            confidence=ev_dict.get("confidence", 1.0),
            meta=ev_dict.get("meta", {}),
        )
        coord.apply_event(ev, bind_to_kernel_moment=False)

    status = coord.get_status()
    return status


def replay_project(atlas_dir: Path, project_slug: str):
    """Replay bytes + events to reconstruct state for a project from .aci/ artifacts."""
    aci_dir = get_aci_dir()
    bytes_path = aci_dir / f"{project_slug}.bytes"
    events_path = aci_dir / f"{project_slug}.events.jsonl"
    return replay_from_logs(atlas_dir, bytes_path, events_path)


def bundle_project(atlas_dir: Path, project_md_path: Path) -> Path:
    """
    Create a bundle for a project.
    Takes the project markdown file path (filename can differ from project_slug).
    Returns path to the created bundle.
    """
    import zipfile
    from datetime import datetime
    
    if not project_md_path.exists():
        raise FileNotFoundError(f"Project file not found: {project_md_path}")
    
    # Read project_slug from markdown (bracket notation format)
    project_slug, _, _, _, _ = parse_project_from_markdown(project_md_path)
    
    aci_dir = get_aci_dir()
    bytes_path = aci_dir / f"{project_slug}.bytes"
    events_path = aci_dir / f"{project_slug}.events.jsonl"
    report_json_path = aci_dir / f"{project_slug}.report.json"
    report_md_path = aci_dir / f"{project_slug}.report.md"
    id_path = aci_dir / f"{project_slug}.id"
    
    # Require compiled artifacts (must come from sync_project, not created here)
    if not bytes_path.exists():
        raise FileNotFoundError(f"Missing compiled bytes: {bytes_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing compiled events: {events_path}")
    if not id_path.exists():
        raise FileNotFoundError(f"Missing project identity: {id_path}")
    
    # Replay to get final state
    status = replay_project(atlas_dir, project_slug)
    
    # Require report artifacts (must come from sync_project, not created here)
    if not report_json_path.exists():
        raise FileNotFoundError(f"Missing compiled report.json: {report_json_path}")
    if not report_md_path.exists():
        raise FileNotFoundError(f"Missing compiled report.md: {report_md_path}")
    
    # Read project_id value (for bundle manifest)
    project_id_value = id_path.read_text(encoding="utf-8").strip()
    
    # Compute hashes
    bytes_hash = file_sha256(bytes_path)
    events_hash = file_sha256(events_path)
    project_hash = file_sha256(project_md_path)
    report_json_hash = file_sha256(report_json_path)
    report_md_hash = file_sha256(report_md_path)
    project_id_hash = file_sha256(id_path)
    
    # Build bundle.json
    bundle_data = {
        "project_slug": project_slug,
        "project_id": project_id_value,
        "byte_seed_version": "AIR_AR_BYTES_V1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "kernel": {
            "step": status.kernel["step"],
            "state_index": status.kernel["state_index"],
            "state_hex": status.kernel["state_hex"],
            "a_hex": status.kernel["a_hex"],
            "b_hex": status.kernel["b_hex"],
            "last_byte": status.kernel["last_byte"],
        },
        "logs": {
            "byte_count": status.kernel["byte_log_len"],
            "event_count": status.kernel["event_log_len"],
            "bytes_sha256": bytes_hash,
            "events_sha256": events_hash,
            "project_md_sha256": project_hash,
            "report_json_sha256": report_json_hash,
            "report_md_sha256": report_md_hash,
            "project_id_sha256": project_id_hash,
        },
        "apertures": {
            "economy": status.apertures["econ"],
            "employment": status.apertures["emp"],
            "education": status.apertures["edu"],
        },
    }
    
    # Bundle goes to bundles_dir/<project_slug>.zip
    bundles_dir = get_bundles_dir()
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_out = bundles_dir / f"{project_slug}.zip"
    
    # Create zip file
    with zipfile.ZipFile(bundle_out, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add project.md
        zf.write(project_md_path, "project.md")
        
        # Add bytes and events files
        zf.write(bytes_path, "bytes.bin")
        zf.write(events_path, "events.jsonl")
        
        # Add reports (required)
        zf.write(report_json_path, "report.json")
        zf.write(report_md_path, "report.md")
        
        # Add project identity file
        zf.write(id_path, "project.id")
        
        # Add bundle.json
        zf.writestr("bundle.json", json.dumps(bundle_data, indent=2))
    
    return bundle_out


def verify_event_bindings(atlas_dir: Path, bytes_path: Path, events_path: Path) -> bool:
    """
    Verify that each event's kernel binding (kernel_step, kernel_state_index, kernel_last_byte)
    matches the actual kernel state at that point in the byte log.
    
    Returns True if all bindings are consistent and all events are checked, False otherwise.
    """
    from src.router.kernel import RouterKernel
    
    # Create a fresh kernel to step through
    kernel = RouterKernel(atlas_dir)
    kernel.reset()
    
    # Read all bytes and events
    byte_list = list(read_bytes(bytes_path))
    events_list = list(read_events(events_path))
    
    max_step = len(byte_list)
    
    # Group events by step and validate step bounds early
    events_by_step: Dict[int, list[Dict[str, Any]]] = {}
    for ev in events_list:
        step = ev.get("kernel_step")
        if not isinstance(step, int):
            return False
        # Reject steps outside valid range [1, max_step]
        if step < 1 or step > max_step:
            return False
        events_by_step.setdefault(step, []).append(ev)
    
    # Step through bytes and check events at each step
    checked_events = 0
    
    for byte_val in byte_list:
        kernel.step_byte(byte_val)
        step = kernel.step
        
        # Check events bound to this step
        for ev in events_by_step.get(step, []):
            if ev.get("kernel_step") != step:
                return False
            
            expected_state_index = ev.get("kernel_state_index")
            expected_last_byte = ev.get("kernel_last_byte")
            if not isinstance(expected_state_index, int) or not isinstance(expected_last_byte, int):
                return False
            
            if kernel.state_index != expected_state_index:
                return False
            if kernel.last_byte != expected_last_byte:
                return False
            
            checked_events += 1
    
    # Ensure we checked every event in the file
    return checked_events == len(events_list)


def verify_bundle(atlas_dir: Path, bundle_path: Path) -> bool:
    """
    Verify a bundle by replaying and checking hashes and state.
    Returns True if verification passes, False otherwise.
    """
    import zipfile
    import tempfile
    
    if not bundle_path.exists():
        return False
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            with zipfile.ZipFile(bundle_path, "r") as zf:
                zf.extractall(tmp_path)
            
            # Read bundle.json
            bundle_json_path = tmp_path / "bundle.json"
            if not bundle_json_path.exists():
                return False
            
            bundle_data = json.loads(bundle_json_path.read_text())
            
            # Check required files
            bytes_file = tmp_path / "bytes.bin"
            events_file = tmp_path / "events.jsonl"
            
            if not bytes_file.exists() or not events_file.exists():
                return False
            
            # Verify event bindings (structural integrity check)
            if not verify_event_bindings(atlas_dir, bytes_file, events_file):
                return False
            
            # Replay from bundle contents
            status = replay_from_logs(atlas_dir, bytes_file, events_file)
            
            # Verify kernel signature (all fields: step, state_index, state_hex, a_hex, b_hex, last_byte)
            kernel_match = (
                status.kernel.get("step") == bundle_data["kernel"].get("step")
                and status.kernel["state_index"] == bundle_data["kernel"]["state_index"]
                and status.kernel["state_hex"] == bundle_data["kernel"]["state_hex"]
                and status.kernel["a_hex"] == bundle_data["kernel"]["a_hex"]
                and status.kernel["b_hex"] == bundle_data["kernel"]["b_hex"]
                and status.kernel["last_byte"] == bundle_data["kernel"]["last_byte"]
            )
            
            if not kernel_match:
                return False
            
            # Verify reports exist in bundle (required)
            report_json_file = tmp_path / "report.json"
            report_md_file = tmp_path / "report.md"
            if not report_json_file.exists() or not report_md_file.exists():
                return False
            
            # Verify report hashes
            report_json_hash = file_sha256(report_json_file)
            report_md_hash = file_sha256(report_md_file)
            
            if "report_json_sha256" in bundle_data["logs"]:
                if report_json_hash != bundle_data["logs"]["report_json_sha256"]:
                    return False
            if "report_md_sha256" in bundle_data["logs"]:
                if report_md_hash != bundle_data["logs"]["report_md_sha256"]:
                    return False
            
            # Verify apertures
            apertures_match = (
                abs(status.apertures["econ"] - bundle_data["apertures"]["economy"]) < 1e-6
                and abs(status.apertures["emp"] - bundle_data["apertures"]["employment"]) < 1e-6
                and abs(status.apertures["edu"] - bundle_data["apertures"]["education"]) < 1e-6
            )
            
            if not apertures_match:
                return False
            
            # Verify hashes
            bytes_hash = file_sha256(bytes_file)
            events_hash = file_sha256(events_file)
            project_md_file = tmp_path / "project.md"
            project_hash = file_sha256(project_md_file) if project_md_file.exists() else ""
            
            hashes_match = (
                bytes_hash == bundle_data["logs"]["bytes_sha256"]
                and events_hash == bundle_data["logs"]["events_sha256"]
            )
            
            if not hashes_match:
                return False
            
            # Verify project.md hash if present in bundle
            if "project_md_sha256" in bundle_data["logs"]:
                if project_hash != bundle_data["logs"]["project_md_sha256"]:
                    return False
            
            # Verify project.id (required, must exist and match value)
            project_id_file = tmp_path / "project.id"
            if not project_id_file.exists():
                return False
            
            project_id_value = project_id_file.read_text(encoding="utf-8").strip()
            if project_id_value != bundle_data.get("project_id", ""):
                return False
            
            # Verify project.id hash if present in bundle
            if "project_id_sha256" in bundle_data["logs"]:
                project_id_hash = file_sha256(project_id_file)
                if project_id_hash != bundle_data["logs"]["project_id_sha256"]:
                    return False
            
            return True
            
    except Exception:
        return False


def ensure_project_id(project_slug: str) -> str:
    """
    Ensure project has a stable project_id stored in .aci/<slug>.id
    If missing, generate one and persist it.
    Returns the project_id.
    """
    import uuid
    
    aci_dir = get_aci_dir()
    id_path = aci_dir / f"{project_slug}.id"
    
    if id_path.exists():
        pid = id_path.read_text(encoding="utf-8").strip()
        if pid:
            # Validate UUID format - raise error if invalid (don't silently regenerate)
            try:
                uuid.UUID(pid)
                return pid
            except ValueError:
                raise ValueError(
                    f"Invalid UUID in project identity file: {id_path}. "
                    "Delete the file to regenerate, or fix the UUID manually."
                )
    
    # Generate new UUID
    new_id = str(uuid.uuid4())
    
    # Persist
    aci_dir.mkdir(parents=True, exist_ok=True)
    id_path.write_text(new_id, encoding="utf-8")
    
    return new_id


def sync_project(atlas_dir: Path, project_md_path: Path) -> Dict[str, Any]:
    """
    Sync a project: parse attestations from project.md, compile kernel log and events.
    Returns summary dict with event_count, apertures, etc.
    
    Attestations are compiled into kernel facts which generate:
    - Kernel log: append-only record of dimensionful transitions (daily/sprint units)
    - Classification ledger: THM-only (for Hodge/aperture accounting)
    - Gyroscope: counted in reports but NOT injected into ledger
    - Report artifacts: .report.json and .report.md
    
    Format: Markdown body with bracket notation (GTD:[5], GMT:[3], etc.)
    Following GGG methodology: all terms are used to sustain balance.
    """
    from src.app.coordination import Coordinator
    from src.plugins.frameworks import THMDisplacementPlugin, PluginContext
    
    # Parse from markdown body with bracket notation
    project_slug, domain_counts, principle_counts, unit, _ = parse_project_from_markdown(project_md_path)
    
    # Check for empty project (all counts are 0)
    total_incidents = sum(principle_counts.values())
    total_domain_count = sum(domain_counts.values())
    
    # Check for parsing warnings
    parse_warnings = []
    if total_incidents > 0 and total_domain_count == 0:
        parse_warnings.append("All domain counts are zero, but incidents are present. All incidents will be distributed evenly across domains.")
    if total_incidents == 0 and total_domain_count > 0:
        parse_warnings.append("Domain counts are present, but no incidents recorded. Project will have zero attestations.")
    
    # Check for potential malformed template (all counts zero but file appears modified)
    if total_incidents == 0 and total_domain_count == 0:
        template_size = len(templates.PROJECT_TEMPLATE_MD.encode("utf-8"))
        file_size = project_md_path.stat().st_size
        if file_size > template_size:
            parse_warnings.append("All counts parsed as zero, but project.md appears modified. Check that bracket notation is exact (e.g., 'Economy: [5]' not 'Economy: [ 5 ]' or 'Economy: [5] // comment').")
    
    # Ensure project has stable ID (stored in .aci/<slug>.id)
    project_id = ensure_project_id(project_slug)
    
    # Generate attestations from counts (following GGG balance principle)
    attestations = generate_attestations_from_counts(domain_counts, principle_counts, project_id, unit)
    
    # Determine artifact paths in .aci/ directory
    aci_dir = get_aci_dir()
    bytes_path = aci_dir / f"{project_slug}.bytes"
    events_path = aci_dir / f"{project_slug}.events.jsonl"
    report_json_path = aci_dir / f"{project_slug}.report.json"
    report_md_path = aci_dir / f"{project_slug}.report.md"
    
    # Initialize coordinator
    coord = Coordinator(atlas_dir)
    
    # Hash-based canonical bytes per attestation
    def canonical_unit_bytes(project_id: str, att_id: str, att_idx: int, unit: str) -> bytes:
        """
        Generate canonical bytes for an attestation using SHA-256 hash.
        Seed format: AIR_AR_BYTES_V1|<project_id>|<attestation_id>|<attestation_index>|<unit>
        """
        seed = f"AIR_AR_BYTES_V1|{project_id}|{att_id}|{att_idx}|{unit}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        if unit == "daily":
            return digest[:1]
        elif unit == "sprint":
            return digest[:4]
        return b""
    
    def unit_weight(unit: str) -> int:
        """Return weight for a unit (for accounting/ledger)."""
        if unit == "daily":
            return 1
        elif unit == "sprint":
            return 4
        else:
            raise ValueError(f"Unknown unit: {unit}")
    
    # Mapping from full names to abbreviations for THM
    thm_map = {
        "governance traceability displacement": "GTD",
        "information variety displacement": "IVD",
        "inference accountability displacement": "IAD",
        "intelligence integrity displacement": "IID",
    }
    
    # Gyroscope categories (for accounting only, not ledger)
    gyro_categories = {
        "governance management traceability": "GMT",
        "information curation variety": "ICV",
        "inference interaction accountability": "IIA",
        "intelligence cooperation integrity": "ICI",
    }
    
    # Build bytes and events in exact order as stepping
    byte_log = bytearray()
    bound_events = []
    
    # Accounting rollups (for reports)
    thm_counts = {"GTD": 0, "IVD": 0, "IAD": 0, "IID": 0}
    thm_by_domain = {"economy": {"GTD": 0, "IVD": 0, "IAD": 0, "IID": 0},
                     "employment": {"GTD": 0, "IVD": 0, "IAD": 0, "IID": 0},
                     "education": {"GTD": 0, "IVD": 0, "IAD": 0, "IID": 0}}
    gyro_counts = {"GMT": 0, "ICV": 0, "IIA": 0, "ICI": 0}
    gyro_by_domain = {"economy": {"GMT": 0, "ICV": 0, "IIA": 0, "ICI": 0},
                     "employment": {"GMT": 0, "ICV": 0, "IIA": 0, "ICI": 0},
                     "education": {"GMT": 0, "ICV": 0, "IIA": 0, "ICI": 0}}
    missing_ids = []
    skipped_attestations = []
    processed_count = 0
    
    # Process each attestation in order
    for att_idx, att in enumerate(attestations):
        # Type guard: attestations from generate_attestations_from_counts are always dicts
        # but we check for safety in case of future changes
        if not isinstance(att, dict):  # type: ignore[redundant-expr]
            skipped_attestations.append({"index": att_idx, "id": None, "reason": "not a dict"})
            continue
        
        unit = att.get("unit", "").lower()
        domain = att.get("domain", "").lower()
        human_mark = att.get("human_mark", "").lower()
        gyroscope_work = att.get("gyroscope_work", "").lower()
        att_id = att.get("id", "")
        
        if unit not in ["daily", "sprint"]:
            skipped_attestations.append({"index": att_idx, "id": att_id or f"att_{att_idx}", "reason": f"invalid unit: {unit}"})
            continue
        if domain not in ["economy", "employment", "education"]:
            skipped_attestations.append({"index": att_idx, "id": att_id or f"att_{att_idx}", "reason": f"invalid domain: {domain}"})
            continue
        
        # Use stable attestation_id, fallback to generated ID
        if not att_id:
            att_id = f"att_{att_idx}"
            missing_ids.append({"index": att_idx, "generated_id": att_id})
        
        processed_count += 1
        
        # Get canonical bytes for this attestation (hash-based)
        unit_bytes = canonical_unit_bytes(project_id, att_id, att_idx, unit)
        unit_wt = unit_weight(unit)  # Weight for accounting/ledger
        
        # Step kernel with these bytes in order
        for b in unit_bytes:
            coord.step_byte(b)
            byte_log.append(b)
        
        # Extract stable identifiers for meta (not entire attestation)
        att_meta = {
            "attestation_id": att_id,
            "unit": unit,
            "domain": domain,
        }
        if human_mark:
            att_meta["human_mark"] = human_mark
        if gyroscope_work:
            att_meta["gyroscope_work"] = gyroscope_work
        
        # Process THM human_mark if provided (ONLY THM goes to ledger)
        if human_mark:
            thm_abbrev = thm_map.get(human_mark)
            if thm_abbrev:
                # Count for accounting (weighted by unit)
                thm_counts[thm_abbrev] += unit_wt
                thm_by_domain[domain][thm_abbrev] += unit_wt
                
                # Emit ledger event (THM-only) with weighted magnitude
                payload = {"domain": domain}
                payload[thm_abbrev] = float(unit_wt)
                payload["confidence"] = 1.0
                
                plugin = THMDisplacementPlugin()
                ctx = PluginContext(meta=att_meta)
                events = plugin.emit_events(payload, ctx)
                
                # Apply events immediately so they bind to the current kernel state
                for event in events:
                    coord.apply_event(event, bind_to_kernel_moment=True)
                    if coord.event_log:
                        event_dict = coord.event_log[-1]["event"]
                        # Strict check: event must have kernel binding fields
                        if event_dict.get("kernel_step") is None:
                            raise RuntimeError(f"Event missing kernel_step binding: {event_dict}")
                        if event_dict.get("kernel_state_index") is None:
                            raise RuntimeError(f"Event missing kernel_state_index binding: {event_dict}")
                        if event_dict.get("kernel_last_byte") is None:
                            raise RuntimeError(f"Event missing kernel_last_byte binding: {event_dict}")
                        bound_events.append(event_dict)
        
        # Process Gyroscope work if provided (accounting only, NO ledger events)
        if gyroscope_work:
            gyro_abbrev = gyro_categories.get(gyroscope_work)
            if gyro_abbrev:
                # Count for accounting (weighted by unit)
                gyro_counts[gyro_abbrev] += unit_wt
                gyro_by_domain[domain][gyro_abbrev] += unit_wt
                # DO NOT emit ledger events - Gyroscope is for reports only
    
    # Write bytes file (exact order as stepped)
    bytes_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_path.write_bytes(bytes(byte_log))
    
    # Write events.jsonl
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w", encoding="utf-8") as f:
        for ev_dict in bound_events:
            f.write(json.dumps(ev_dict) + "\n")
    
    # Get final status
    status = coord.get_status()
    
    # Verify invariant: step == byte_log_len
    if status.kernel["step"] != status.kernel["byte_log_len"]:
        raise RuntimeError(f"Invariant violation: step ({status.kernel['step']}) != byte_log_len ({status.kernel['byte_log_len']})")
    
    # Compute hashes
    bytes_hash = file_sha256(bytes_path)
    events_hash = file_sha256(events_path)
    
    # Generate report
    report_data = {
        "project_slug": project_slug,
        "project_id": project_id,
        "compilation": {
            "attestation_count": len(attestations),
            "processed_attestations": processed_count,
            "skipped_attestations": skipped_attestations,
            "byte_count": len(byte_log),
            "kernel": {
                "step": status.kernel["step"],
                "state_index": status.kernel["state_index"],
                "state_hex": status.kernel["state_hex"],
                "a_hex": status.kernel["a_hex"],
                "b_hex": status.kernel["b_hex"],
                "last_byte": status.kernel["last_byte"],
            },
            "hashes": {
                "bytes_sha256": bytes_hash,
                "events_sha256": events_hash,
            },
        },
        "accounting": {
            "thm": {
                "totals": thm_counts,
                "by_domain": thm_by_domain,
            },
            "gyroscope": {
                "totals": gyro_counts,
                "by_domain": gyro_by_domain,
            },
        },
        "ledger": {
            "y_econ": status.ledgers["y_econ"],
            "y_emp": status.ledgers["y_emp"],
            "y_edu": status.ledgers["y_edu"],
        },
        "apertures": {
            "A_econ": status.apertures["econ"],
            "A_emp": status.apertures["emp"],
            "A_edu": status.apertures["edu"],
        },
        "warnings": {
            "missing_attestation_ids": missing_ids,
            "empty_project": total_incidents == 0,
            "parse_warnings": parse_warnings,
        } if (missing_ids or total_incidents == 0 or parse_warnings) else {},
    }
    
    # Write report.json
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    
    # Write report.md
    report_md_lines = [
        f"# Project Report: {project_slug}",
        "",
        f"Project ID: {project_id}",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "## Compilation",
        "",
        f"- Total attestations: {len(attestations)}",
        f"- Processed: {processed_count}",
        f"- Skipped: {len(skipped_attestations)}",
        f"- Bytes: {len(byte_log)}",
        f"- Kernel step: {status.kernel['step']}",
        f"- Kernel state: {status.kernel['state_hex']} (index {status.kernel['state_index']})",
        f"- Last byte: 0x{status.kernel['last_byte']:02x}",
        "",
    ]
    
    if skipped_attestations:
        report_md_lines.append("### Skipped Attestations\n")
        for skipped in skipped_attestations:
            report_md_lines.append(f"- Index {skipped['index']} (ID: {skipped['id']}): {skipped['reason']}")
        report_md_lines.append("")
    
    report_md_lines.extend([
        "## Accounting",
        "",
        "### THM Totals",
        f"- GTD: {thm_counts['GTD']}",
        f"- IVD: {thm_counts['IVD']}",
        f"- IAD: {thm_counts['IAD']}",
        f"- IID: {thm_counts['IID']}",
        "",
        "### Gyroscope Totals",
        f"- GMT: {gyro_counts['GMT']}",
        f"- ICV: {gyro_counts['ICV']}",
        f"- IIA: {gyro_counts['IIA']}",
        f"- ICI: {gyro_counts['ICI']}",
        "",
        "### Attestation Distribution by Domain",
        "",
    ])
    
    # Calculate distribution totals by domain
    for domain in ["economy", "employment", "education"]:
        domain_thm_total = sum(thm_by_domain[domain].values())
        domain_gyro_total = sum(gyro_by_domain[domain].values())
        report_md_lines.append(f"- {domain.capitalize()}: {domain_thm_total} displacement, {domain_gyro_total} alignment")
    
    report_md_lines.extend([
        "",
        "## Ledger & Apertures",
        "",
        f"- Economy aperture: {status.apertures['econ']:.6f}",
        f"- Employment aperture: {status.apertures['emp']:.6f}",
        f"- Education aperture: {status.apertures['edu']:.6f}",
        "",
    ])
    
    if missing_ids or total_incidents == 0 or parse_warnings:
        report_md_lines.append("## Warnings\n")
        if total_incidents == 0:
            report_md_lines.append("- Empty project: No incidents recorded. All bracket counts are 0.\n")
        for warning in parse_warnings:
            report_md_lines.append(f"- {warning}\n")
        for missing in missing_ids:
            report_md_lines.append(f"- Index {missing['index']}: Generated ID `{missing['generated_id']}` (original ID missing)\n")
    
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(report_md_lines), encoding="utf-8")
    
    return {
        "event_count": len(bound_events),
        "apertures": {
            "economy": status.apertures["econ"],
            "employment": status.apertures["emp"],
            "education": status.apertures["edu"],
        },
        "kernel": {
            "step": status.kernel["step"],
            "state_index": status.kernel["state_index"],
            "state_hex": status.kernel["state_hex"],
        },
    }

