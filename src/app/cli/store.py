"""
Workspace I/O and replay functions.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from src.app.cli import templates


def get_data_dir() -> Path:
    """Returns data/ directory in current working directory."""
    data_dir = Path.cwd() / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_programs_dir() -> Path:
    """Returns data/programs/ directory."""
    programs_dir = get_data_dir() / "programs"
    if not programs_dir.exists():
        programs_dir.mkdir(parents=True, exist_ok=True)
    return programs_dir


def get_aci_dir() -> Path:
    """Returns data/programs/.aci/ directory for compiled artifacts."""
    aci_dir = get_programs_dir() / ".aci"
    if not aci_dir.exists():
        aci_dir.mkdir(parents=True, exist_ok=True)
    return aci_dir


def get_bundles_dir() -> Path:
    """Returns data/programs/bundles/ directory."""
    bundles_dir = get_programs_dir() / "bundles"
    if not bundles_dir.exists():
        bundles_dir.mkdir(parents=True, exist_ok=True)
    return bundles_dir


def ensure_workspace() -> None:
    """Ensure all workspace directories exist. Called on CLI startup."""
    get_data_dir()
    get_programs_dir()
    get_aci_dir()
    get_bundles_dir()


def ensure_templates() -> None:
    """
    Ensure program template is available in programs directory.
    Template is named with underscore prefix so it's not processed as a program.
    Always overwrites the template to keep it in sync with the code version.
    """
    programs_dir = get_programs_dir()
    program_template = programs_dir / "_template.md"
    program_template.write_text(templates.PROGRAM_TEMPLATE_MD, encoding="utf-8")


def parse_bracket_value(text: str, pattern: str) -> int:
    """
    Parse a bracket value from text using pattern.
    Pattern should contain a capture group for the value.
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
    notes_pattern = r'^##\s+NOTES\s*\n---?\s*\n(.*?)(?=^##|\Z)'
    match = re.search(notes_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if match:
        notes = match.group(1).strip()
        if notes == "(Add context or key observations for this program)":
            return ""
        return notes
    return ""


def parse_participants_section(text: str) -> tuple[str, str]:
    """
    Parse the PARTICIPANTS section from markdown text.
    Returns: (agents, agencies)
    """
    agents_pattern = r'###\s+Agents\s*\n+(.*?)(?=###|^##|\Z)'
    agents_match = re.search(agents_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    agents = ""
    if agents_match:
        agents = agents_match.group(1).strip()
        placeholder = "(Names of people involved in this program)"
        if agents == placeholder or agents.strip() == placeholder:
            agents = ""

    agencies_pattern = r'###\s+Agencies\s*\n+(.*?)(?=^---|^##|\Z)'
    agencies_match = re.search(agencies_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    agencies = ""
    if agencies_match:
        agencies = agencies_match.group(1).strip()
        placeholder = "(Names of agencies involved in this program)"
        if agencies == placeholder or agencies.strip() == placeholder:
            agencies = ""

    return agents, agencies


def parse_program_from_markdown(program_md_path: Path) -> tuple[str, dict[str, int], dict[str, int], str, str, str, str]:
    """
    Parse program from markdown body using bracket notation.
    Returns: (program_slug, domain_counts, principle_counts, unit, notes, agents, agencies)
    """
    text = program_md_path.read_text(encoding="utf-8")

    program_slug = program_md_path.stem

    domain_counts = {
        "economy": parse_bracket_value(text, r'Economy[^:]*:\s*\[(\d+)\]'),
        "employment": parse_bracket_value(text, r'Employment[^:]*:\s*\[(\d+)\]'),
        "education": parse_bracket_value(text, r'Education[^:]*:\s*\[(\d+)\]'),
    }

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

    unit_match = re.search(r'Unit:\s*\[(daily|sprint)\]', text, re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(1).lower()
        if unit not in ["daily", "sprint"]:
            unit = "daily"
    else:
        unit = "daily"

    notes = parse_notes_section(text)
    agents, agencies = parse_participants_section(text)

    return program_slug, domain_counts, principle_counts, unit, notes, agents, agencies


def generate_attestations_from_counts(
    domain_counts: dict[str, int],
    principle_counts: dict[str, int],
    program_id: str,
    unit: str = "daily"
) -> list[dict[str, Any]]:
    """
    Generate attestations from domain and principle counts.
    Following GGG_Methodology: all terms are used to sustain balance.
    """
    attestations = []

    thm_displacement_map = {
        "GTD": "governance traceability displacement",
        "IVD": "information variety displacement",
        "IAD": "inference accountability displacement",
        "IID": "intelligence integrity displacement",
    }

    gyro_alignment_map = {
        "GMT": "governance management traceability",
        "ICV": "information curation variety",
        "IIA": "inference interaction accountability",
        "ICI": "intelligence cooperation integrity",
    }

    total_domain_count = sum(domain_counts.values())
    domains = ["economy", "employment", "education"]

    domain_weights = {}
    for domain in domains:
        domain_weights[domain] = domain_counts.get(domain, 0)

    if total_domain_count == 0:
        for domain in domains:
            domain_weights[domain] = 1
        total_domain_count = 3

    def select_domain_proportional(incident_idx: int, total_incidents: int) -> str:
        if total_incidents == 0:
            for domain in domains:
                if domain_weights[domain] > 0:
                    return domain
            return domains[0]

        cumulative = 0
        thresholds = []
        for domain in domains:
            cumulative += domain_weights[domain]
            thresholds.append((cumulative, domain))

        position = (incident_idx * total_domain_count) // total_incidents
        position = position % total_domain_count

        for threshold, domain in thresholds:
            if position < threshold:
                return domain

        for domain in domains:
            if domain_weights[domain] > 0:
                return domain
        return domains[0]

    att_idx = 0
    total_displacement_incidents = sum(principle_counts.get(abbrev, 0) for abbrev in thm_displacement_map.keys())

    for abbrev, full_name in thm_displacement_map.items():
        count = principle_counts.get(abbrev, 0)
        for i in range(count):
            domain = select_domain_proportional(att_idx, total_displacement_incidents)
            attestations.append({
                "id": f"{abbrev.lower()}_{i+1}",
                "unit": unit,
                "domain": domain,
                "human_mark": full_name,
            })
            att_idx += 1

    total_alignment_incidents = sum(principle_counts.get(abbrev, 0) for abbrev in gyro_alignment_map.keys())
    alignment_att_idx = 0

    for abbrev, full_name in gyro_alignment_map.items():
        count = principle_counts.get(abbrev, 0)
        for i in range(count):
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


def derive_domain_counts_from_events(events_path: Path) -> dict[str, int]:
    """Derive domain_counts from an events.jsonl file."""
    from src.app.events import Domain

    counts = {
        "economy": 0,
        "employment": 0,
        "education": 0,
    }

    if not events_path.exists():
        return counts

    for ev_dict in read_events(events_path):
        domain_int = ev_dict.get("domain")
        if domain_int is not None:
            domain = Domain(domain_int)
            if domain == Domain.ECONOMY:
                counts["economy"] += 1
            elif domain == Domain.EMPLOYMENT:
                counts["employment"] += 1
            elif domain == Domain.EDUCATION:
                counts["education"] += 1

    return counts


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


def replay_from_logs(bytes_path: Path, events_path: Path):
    """Replay bytes + events from given file paths to reconstruct state."""
    from src.app.coordination import Coordinator

    coord = Coordinator()

    for b in read_bytes(bytes_path):
        coord.step_byte(b)

    from src.tools.main.api import event_from_dict
    for ev_dict in read_events(events_path):
        ev = event_from_dict(ev_dict)
        coord.apply_event(ev, bind_to_kernel_moment=False)

    status = coord.get_status()
    return status


def replay_program(program_slug: str):
    """Replay bytes + events to reconstruct state for a program from .aci/ artifacts."""
    aci_dir = get_aci_dir()
    bytes_path = aci_dir / f"{program_slug}.bytes"
    events_path = aci_dir / f"{program_slug}.events.jsonl"
    return replay_from_logs(bytes_path, events_path)


def bundle_program(program_md_path: Path, private_key: Any = None) -> Path:
    """
    Create a bundle for a program.
    Returns path to the created bundle.
    """
    import zipfile

    if not program_md_path.exists():
        raise FileNotFoundError(f"Program file not found: {program_md_path}")

    program_slug, _, _, _, _, _, _ = parse_program_from_markdown(program_md_path)

    aci_dir = get_aci_dir()
    bytes_path = aci_dir / f"{program_slug}.bytes"
    events_path = aci_dir / f"{program_slug}.events.jsonl"
    report_json_path = aci_dir / f"{program_slug}.report.json"
    report_md_path = aci_dir / f"{program_slug}.report.md"
    id_path = aci_dir / f"{program_slug}.id"

    if not bytes_path.exists():
        raise FileNotFoundError(f"Missing compiled bytes: {bytes_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing compiled events: {events_path}")
    if not id_path.exists():
        raise FileNotFoundError(f"Missing program identity: {id_path}")

    status = replay_program(program_slug)

    if not report_json_path.exists():
        raise FileNotFoundError(f"Missing compiled report.json: {report_json_path}")
    if not report_md_path.exists():
        raise FileNotFoundError(f"Missing compiled report.md: {report_md_path}")

    program_id_value = id_path.read_text(encoding="utf-8").strip()

    bytes_hash = file_sha256(bytes_path)
    events_hash = file_sha256(events_path)
    program_hash = file_sha256(program_md_path)
    report_json_hash = file_sha256(report_json_path)
    report_md_hash = file_sha256(report_md_path)
    program_id_hash = file_sha256(id_path)

    grants_path = aci_dir / f"{program_slug}.grants.jsonl"
    shells_path = aci_dir / f"{program_slug}.shells.jsonl"
    archive_path = aci_dir / f"{program_slug}.archive.json"
    grants_hash = file_sha256(grants_path) if grants_path.exists() else ""
    shells_hash = file_sha256(shells_path) if shells_path.exists() else ""
    archive_hash = file_sha256(archive_path) if archive_path.exists() else ""

    meta_root = ""
    if shells_path.exists():
        try:
            from src.app.coordination import Coordinator
            coord_temp = Coordinator()
            shells_data = []
            with open(shells_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        shells_data.append(json.loads(line))
            if shells_data:
                seals = [s["seal"] for s in shells_data]
                meta_root = coord_temp.meta_root_from_shell_seals(seals)
        except Exception:
            pass

    bundle_data = {
        "program_slug": program_slug,
        "program_id": program_id_value,
        "byte_seed_version": "AIR_AR_BYTES_V1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "kernel": {
            "step": status.kernel["step"],
            "state24": status.kernel["state24"],
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
            "program_md_sha256": program_hash,
            "report_json_sha256": report_json_hash,
            "report_md_sha256": report_md_hash,
            "program_id_sha256": program_id_hash,
        },
        "apertures": {
            "economy": status.apertures["econ"],
            "employment": status.apertures["emp"],
            "education": status.apertures["edu"],
        },
    }

    if grants_hash or shells_hash or archive_hash:
        bundle_data["ecology"] = {
            "grants_sha256": grants_hash,
            "shells_sha256": shells_hash,
            "archive_sha256": archive_hash,
            "meta_root": meta_root,
        }

    bundle_json_bytes = json.dumps(bundle_data, indent=2).encode("utf-8")

    if private_key is not None:
        from src.app.coordination import Coordinator
        coord = Coordinator()
        signature = coord.sign_bundle(bundle_json_bytes, private_key)
        signature_hex = signature.hex()

        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes_raw()
        public_key_hex = public_key_bytes.hex()

        bundle_data["signer_public_key"] = public_key_hex
        bundle_data["signature"] = signature_hex

    bundles_dir = get_bundles_dir()
    bundles_dir.mkdir(parents=True, exist_ok=True)
    bundle_out = bundles_dir / f"{program_slug}.zip"

    with zipfile.ZipFile(bundle_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(program_md_path, "program.md")
        zf.write(bytes_path, "bytes.bin")
        zf.write(events_path, "events.jsonl")
        zf.write(report_json_path, "report.json")
        zf.write(report_md_path, "report.md")
        zf.write(id_path, "program.id")

        if grants_path.exists():
            zf.write(grants_path, "grants.jsonl")
        if shells_path.exists():
            zf.write(shells_path, "shells.jsonl")
        if archive_path.exists():
            zf.write(archive_path, "archive.json")

        zf.writestr("bundle.json", json.dumps(bundle_data, indent=2))

    return bundle_out


def verify_event_bindings(bytes_path: Path, events_path: Path) -> bool:
    """
    Verify that each event's kernel binding (kernel_step, kernel_state24, kernel_last_byte)
    matches the actual kernel state at that point in the byte log.

    Returns True if all bindings are consistent, False otherwise.
    """
    from src.kernel import Gyroscopic

    kernel = Gyroscopic()
    kernel.reset()

    byte_list = list(read_bytes(bytes_path))
    events_list = list(read_events(events_path))

    max_step = len(byte_list)

    events_by_step: dict[int, list[dict[str, Any]]] = {}
    for ev in events_list:
        step = ev.get("kernel_step")
        if not isinstance(step, int):
            return False
        if step < 1 or step > max_step:
            return False
        events_by_step.setdefault(step, []).append(ev)

    checked_events = 0

    for byte_val in byte_list:
        kernel.step_byte(byte_val)
        step = kernel.step

        for ev in events_by_step.get(step, []):
            if ev.get("kernel_step") != step:
                return False

            expected_state24 = ev.get("kernel_state24")
            expected_last_byte = ev.get("kernel_last_byte")
            if not isinstance(expected_state24, int) or not isinstance(expected_last_byte, int):
                return False

            if kernel.state24 != expected_state24:
                return False
            if kernel.last_byte != expected_last_byte:
                return False

            checked_events += 1

    return checked_events == len(events_list)


def verify_bundle(bundle_path: Path) -> bool:
    """
    Verify a bundle by replaying and checking hashes and state.
    Returns True if verification passes, False otherwise.
    """
    import tempfile
    import zipfile

    if not bundle_path.exists():
        return False

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            with zipfile.ZipFile(bundle_path, "r") as zf:
                zf.extractall(tmp_path)

            bundle_json_path = tmp_path / "bundle.json"
            if not bundle_json_path.exists():
                return False

            bundle_data = json.loads(bundle_json_path.read_text())

            bytes_file = tmp_path / "bytes.bin"
            events_file = tmp_path / "events.jsonl"

            if not bytes_file.exists() or not events_file.exists():
                return False

            if not verify_event_bindings(bytes_file, events_file):
                return False

            status = replay_from_logs(bytes_file, events_file)

            kernel_match = (
                status.kernel.get("step") == bundle_data["kernel"].get("step")
                and status.kernel["state24"] == bundle_data["kernel"]["state24"]
                and status.kernel["state_hex"] == bundle_data["kernel"]["state_hex"]
                and status.kernel["a_hex"] == bundle_data["kernel"]["a_hex"]
                and status.kernel["b_hex"] == bundle_data["kernel"]["b_hex"]
                and status.kernel["last_byte"] == bundle_data["kernel"]["last_byte"]
            )

            if not kernel_match:
                return False

            report_json_file = tmp_path / "report.json"
            report_md_file = tmp_path / "report.md"
            if not report_json_file.exists() or not report_md_file.exists():
                return False

            report_json_hash = file_sha256(report_json_file)
            report_md_hash = file_sha256(report_md_file)

            if "report_json_sha256" in bundle_data["logs"]:
                if report_json_hash != bundle_data["logs"]["report_json_sha256"]:
                    return False
            if "report_md_sha256" in bundle_data["logs"]:
                if report_md_hash != bundle_data["logs"]["report_md_sha256"]:
                    return False

            apertures_match = (
                abs(status.apertures["econ"] - bundle_data["apertures"]["economy"]) < 1e-6
                and abs(status.apertures["emp"] - bundle_data["apertures"]["employment"]) < 1e-6
                and abs(status.apertures["edu"] - bundle_data["apertures"]["education"]) < 1e-6
            )

            if not apertures_match:
                return False

            bytes_hash = file_sha256(bytes_file)
            events_hash = file_sha256(events_file)
            program_md_file = tmp_path / "program.md"
            program_hash = file_sha256(program_md_file) if program_md_file.exists() else ""

            hashes_match = (
                bytes_hash == bundle_data["logs"]["bytes_sha256"]
                and events_hash == bundle_data["logs"]["events_sha256"]
            )

            if not hashes_match:
                return False

            if "program_md_sha256" in bundle_data["logs"]:
                if program_hash != bundle_data["logs"]["program_md_sha256"]:
                    return False

            program_id_file = tmp_path / "program.id"
            if not program_id_file.exists():
                return False

            program_id_value = program_id_file.read_text(encoding="utf-8").strip()
            if program_id_value != bundle_data.get("program_id", ""):
                return False

            if "program_id_sha256" in bundle_data["logs"]:
                program_id_hash = file_sha256(program_id_file)
                if program_id_hash != bundle_data["logs"]["program_id_sha256"]:
                    return False

            if "signature" in bundle_data and "signer_public_key" in bundle_data:
                from src.app.coordination import Coordinator

                signature_hex = bundle_data.pop("signature")
                public_key_hex = bundle_data.pop("signer_public_key")

                bundle_json_bytes = json.dumps(bundle_data, indent=2).encode("utf-8")

                public_key_bytes = bytes.fromhex(public_key_hex)
                public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)

                signature = bytes.fromhex(signature_hex)

                coord = Coordinator()
                if not coord.verify_bundle_signature(bundle_json_bytes, signature, public_key):
                    return False

                bundle_data["signer_public_key"] = public_key_hex
                bundle_data["signature"] = signature_hex

            if "ecology" in bundle_data:
                ecology_data = bundle_data["ecology"]

                grants_file = tmp_path / "grants.jsonl"
                shells_file = tmp_path / "shells.jsonl"
                archive_file = tmp_path / "archive.json"

                if "grants_sha256" in ecology_data and ecology_data["grants_sha256"]:
                    if not grants_file.exists():
                        return False
                    grants_hash = file_sha256(grants_file)
                    if grants_hash != ecology_data["grants_sha256"]:
                        return False

                if "shells_sha256" in ecology_data and ecology_data["shells_sha256"]:
                    if not shells_file.exists():
                        return False
                    shells_hash = file_sha256(shells_file)
                    if shells_hash != ecology_data["shells_sha256"]:
                        return False

                if "archive_sha256" in ecology_data and ecology_data["archive_sha256"]:
                    if not archive_file.exists():
                        return False
                    archive_hash = file_sha256(archive_file)
                    if archive_hash != ecology_data["archive_sha256"]:
                        return False

                if grants_file.exists() and shells_file.exists() and archive_file.exists():
                    from src.app.coordination import Coordinator
                    from src.app.events import Shell

                    coord_eco = Coordinator()

                    with open(shells_file, encoding="utf-8") as f:
                        shells_replayed = []
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            shell_dict = json.loads(line)
                            shells_replayed.append(Shell(**shell_dict))

                    if "meta_root" in ecology_data and ecology_data["meta_root"]:
                        seals = [s.seal for s in shells_replayed]
                        expected_meta_root = coord_eco.meta_root_from_shell_seals(seals)
                        if expected_meta_root != ecology_data["meta_root"]:
                            return False

            return True

    except Exception:
        return False


def ensure_program_id(program_slug: str) -> str:
    """
    Ensure program has a stable program_id stored in .aci/<slug>.id
    """
    import uuid

    aci_dir = get_aci_dir()
    id_path = aci_dir / f"{program_slug}.id"

    if id_path.exists():
        pid = id_path.read_text(encoding="utf-8").strip()
        if pid:
            try:
                uuid.UUID(pid)
                return pid
            except ValueError:
                raise ValueError(
                    f"Invalid UUID in program identity file: {id_path}. "
                    "Delete the file to regenerate, or fix the UUID manually."
                )

    new_id = str(uuid.uuid4())
    aci_dir.mkdir(parents=True, exist_ok=True)
    id_path.write_text(new_id, encoding="utf-8")

    return new_id


def sync_program(program_md_path: Path) -> dict[str, Any]:
    """
    Sync a program: parse attestations from program.md, compile kernel log and events.
    Returns summary dict with event_count, apertures, etc.
    """
    from src.app.coordination import Coordinator
    from src.tools.main.frameworks import ToolContext, THMDisplacementTool

    program_slug, domain_counts, principle_counts, unit, _, agents, agencies = parse_program_from_markdown(program_md_path)

    total_incidents = sum(principle_counts.values())
    total_domain_count = sum(domain_counts.values())

    parse_warnings = []
    if total_incidents > 0 and total_domain_count == 0:
        parse_warnings.append("All domain counts are zero, but incidents are present. All incidents will be distributed evenly across domains.")
    if total_incidents == 0 and total_domain_count > 0:
        parse_warnings.append("Domain counts are present, but no incidents recorded. Program will have zero attestations.")

    if total_incidents == 0 and total_domain_count == 0:
        template_size = len(templates.PROGRAM_TEMPLATE_MD.encode("utf-8"))
        file_size = program_md_path.stat().st_size
        if file_size > template_size:
            parse_warnings.append("All counts parsed as zero, but program.md appears modified. Check that bracket notation is exact (e.g., 'Economy: [5]' not 'Economy: [ 5 ]' or 'Economy: [5] // comment').")

    program_id = ensure_program_id(program_slug)
    attestations = generate_attestations_from_counts(domain_counts, principle_counts, program_id, unit)

    aci_dir = get_aci_dir()
    bytes_path = aci_dir / f"{program_slug}.bytes"
    events_path = aci_dir / f"{program_slug}.events.jsonl"
    report_json_path = aci_dir / f"{program_slug}.report.json"
    report_md_path = aci_dir / f"{program_slug}.report.md"

    coord = Coordinator()

    def canonical_unit_bytes(program_id: str, att_id: str, att_idx: int, unit: str) -> bytes:
        seed = f"AIR_AR_BYTES_V1|{program_id}|{att_id}|{att_idx}|{unit}".encode()
        digest = hashlib.sha256(seed).digest()
        if unit == "daily":
            return digest[:1]
        elif unit == "sprint":
            return digest[:4]
        return b""

    def unit_weight(unit: str) -> int:
        if unit == "daily":
            return 1
        elif unit == "sprint":
            return 4
        else:
            raise ValueError(f"Unknown unit: {unit}")

    thm_map = {
        "governance traceability displacement": "GTD",
        "information variety displacement": "IVD",
        "inference accountability displacement": "IAD",
        "intelligence integrity displacement": "IID",
    }

    gyro_categories = {
        "governance management traceability": "GMT",
        "information curation variety": "ICV",
        "inference interaction accountability": "IIA",
        "intelligence cooperation integrity": "ICI",
    }

    byte_log = bytearray()
    bound_events = []

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

    for att_idx, att in enumerate(attestations):
        if not isinstance(att, dict):
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

        if not att_id:
            att_id = f"att_{att_idx}"
            missing_ids.append({"index": att_idx, "generated_id": att_id})

        processed_count += 1

        unit_bytes = canonical_unit_bytes(program_id, att_id, att_idx, unit)
        unit_wt = unit_weight(unit)

        for b in unit_bytes:
            coord.step_byte(b)
            byte_log.append(b)

        att_meta = {
            "attestation_id": att_id,
            "unit": unit,
            "domain": domain,
        }
        if human_mark:
            att_meta["human_mark"] = human_mark
        if gyroscope_work:
            att_meta["gyroscope_work"] = gyroscope_work

        if human_mark:
            thm_abbrev = thm_map.get(human_mark)
            if thm_abbrev:
                thm_counts[thm_abbrev] += unit_wt
                thm_by_domain[domain][thm_abbrev] += unit_wt

                payload = {thm_abbrev: float(unit_wt), "confidence": 1.0}

                tool = THMDisplacementTool()
                ctx = ToolContext(meta=att_meta)
                events = tool.emit_events(payload, ctx)

                for event in events:
                    coord.apply_event(event, bind_to_kernel_moment=True)
                    if coord.event_log:
                        event_dict = coord.event_log[-1]["event"]
                        if event_dict.get("kernel_step") is None:
                            raise RuntimeError(f"Event missing kernel_step binding: {event_dict}")
                        if event_dict.get("kernel_state24") is None:
                            raise RuntimeError(f"Event missing kernel_state24 binding: {event_dict}")
                        if event_dict.get("kernel_last_byte") is None:
                            raise RuntimeError(f"Event missing kernel_last_byte binding: {event_dict}")
                        bound_events.append(event_dict)

        if gyroscope_work:
            gyro_abbrev = gyro_categories.get(gyroscope_work)
            if gyro_abbrev:
                gyro_counts[gyro_abbrev] += unit_wt
                gyro_by_domain[domain][gyro_abbrev] += unit_wt

    bytes_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_path.write_bytes(bytes(byte_log))

    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w", encoding="utf-8") as f:
        for ev_dict in bound_events:
            f.write(json.dumps(ev_dict) + "\n")

    status = coord.get_status()

    if status.kernel["step"] != status.kernel["byte_log_len"]:
        raise RuntimeError(f"Invariant violation: step ({status.kernel['step']}) != byte_log_len ({status.kernel['byte_log_len']})")

    bytes_hash = file_sha256(bytes_path)
    events_hash = file_sha256(events_path)

    report_data = {
        "program_slug": program_slug,
        "program_id": program_id,
        "compilation": {
            "attestation_count": len(attestations),
            "processed_attestations": processed_count,
            "skipped_attestations": skipped_attestations,
            "byte_count": len(byte_log),
            "kernel": {
                "step": status.kernel["step"],
                "state24": status.kernel["state24"],
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
            "empty_program": total_incidents == 0,
            "parse_warnings": parse_warnings,
        } if (missing_ids or total_incidents == 0 or parse_warnings) else {},
    }

    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    grants_path = aci_dir / f"{program_slug}.grants.jsonl"
    shells_path = aci_dir / f"{program_slug}.shells.jsonl"
    archive_path = aci_dir / f"{program_slug}.archive.json"

    grants_path.parent.mkdir(parents=True, exist_ok=True)
    with open(grants_path, "w", encoding="utf-8") as f:
        for shell, grants_list in coord.fiat_shell_grants_log:
            for grant in grants_list:
                grant_dict = grant.as_dict()
                grant_dict["shell_header"] = shell.header
                f.write(json.dumps(grant_dict) + "\n")

    shells_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shells_path, "w", encoding="utf-8") as f:
        for shell in coord.fiat_shell_log:
            f.write(json.dumps(shell.as_dict()) + "\n")

    from src.app.events import Archive
    archive = Archive(
        per_identity_MU=coord.fiat_status()["per_identity_totals"],
        total_capacity_MU=coord.fiat_capacity_total,
        used_capacity_MU=coord.fiat_used_total,
        free_capacity_MU=coord.fiat_capacity_total - coord.fiat_used_total,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(archive.as_dict(), indent=2), encoding="utf-8")

    report_md_lines = [
        f"# Program Report: {program_slug}",
        "",
        f"Program ID: {program_id}",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
    ]

    if agents or agencies:
        report_md_lines.extend([
            "## Participants",
            "",
        ])
        if agents:
            report_md_lines.extend([
                "### Agents",
                "",
                agents,
                "",
            ])
        if agencies:
            report_md_lines.extend([
                "### Agencies",
                "",
                agencies,
                "",
            ])

    report_md_lines.extend([
        "## Common Source Consensus",
        "",
        "All Artificial categories of Authority and Agency are Indirect",
        "originating from Human Intelligence.",
        "",
    ])

    report_md_lines.extend([
        "## Compilation",
        "",
        f"- Total attestations: {len(attestations)}",
        f"- Processed: {processed_count}",
        f"- Skipped: {len(skipped_attestations)}",
        f"- Bytes: {len(byte_log)}",
        f"- Kernel step: {status.kernel['step']}",
        f"- Kernel state: {status.kernel['state_hex']} (state24: 0x{status.kernel['state24']:06x})",
        f"- Last byte: 0x{status.kernel['last_byte']:02x}",
        "",
    ])

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
            report_md_lines.append("- Empty program: No incidents recorded. All bracket counts are 0.\n")
        for warning in parse_warnings:
            report_md_lines.append(f"- {warning}\n")
        for missing in missing_ids:
            report_md_lines.append(f"- Index {missing['index']}: Generated ID `{missing['generated_id']}` (direct ID missing)\n")

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
            "state24": status.kernel["state24"],
            "state_hex": status.kernel["state_hex"],
        },
    }