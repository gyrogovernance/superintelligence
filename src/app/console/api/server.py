"""
AIR Governance Console API Server.

A thin HTTP layer that imports and calls existing modules.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any
import json
import re
from datetime import datetime

from src.app.cli import store, templates
from src.app.cli.schemas import A_STAR

app = FastAPI(title="AIR Console", version="1.0.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Path helpers
def programs_dir() -> Path:
    return store.get_programs_dir()


def aci_dir() -> Path:
    return store.get_aci_dir()


def bundles_dir() -> Path:
    return store.get_bundles_dir()


def atlas_dir() -> Path:
    return store.get_atlas_dir()


# Governance helpers
def get_bundle_status(slug: str) -> tuple[str, bool, str | None]:
    """
    Get bundle status for a program.
    Returns: (status, signed, signer_fingerprint)
    status: "Local" | "Published" | "Verified"
    signed: True if bundle has valid signature
    signer_fingerprint: First/last 8 hex chars of public key, or None
    """
    bundle_path = bundles_dir() / f"{slug}.zip"
    report_path = aci_dir() / f"{slug}.report.json"
    
    # Check if report exists (Local status minimum)
    if not report_path.exists():
        return ("Local", False, None)
    
    # Report exists but no bundle yet
    if not bundle_path.exists():
        return ("Local", False, None)
    
    # Check signature in bundle
    signed = False
    signer_fingerprint = None
    
    try:
        import zipfile
        with zipfile.ZipFile(bundle_path, "r") as zf:
            if "bundle.json" in zf.namelist():
                bundle_data = json.loads(zf.read("bundle.json").decode("utf-8"))
                if "signer_public_key" in bundle_data and "signature" in bundle_data:
                    signed = True
                    public_key_hex = bundle_data["signer_public_key"]
                    if len(public_key_hex) >= 16:
                        signer_fingerprint = f"{public_key_hex[:8]}...{public_key_hex[-8:]}"
    except Exception:
        pass
    
    # Check if bundle is verified (try verification)
    verified = False
    try:
        verified = store.verify_bundle(atlas_dir(), bundle_path)
    except Exception:
        pass
    
    if verified:
        return ("Verified", signed, signer_fingerprint)
    else:
        return ("Published", signed, signer_fingerprint)


def get_sign_bundle_on_sync() -> bool:
    """Get 'sign bundle on sync' setting from config file."""
    config_path = aci_dir() / ".config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return config.get("sign_bundle_on_sync", False)
        except Exception:
            return False
    return False


def set_sign_bundle_on_sync(value: bool) -> None:
    """Set 'sign bundle on sync' setting in config file."""
    config_path = aci_dir() / ".config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    config["sign_bundle_on_sync"] = value
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _has_private_key() -> bool:
    """Check if a signing key exists in config."""
    config_path = aci_dir() / ".config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return "signing_key_pem" in config
        except Exception:
            return False
    return False


def get_ecology_data(slug: str) -> tuple[Dict[str, Any] | None, list[Dict[str, Any]]]:
    """
    Get ecology data for a program.
    Returns: (archive_data, shells_list)
    archive_data: Archive summary dict or None if no archive
    shells_list: List of Shell dicts (empty if no shells)
    """
    archive_path = aci_dir() / f"{slug}.archive.json"
    shells_path = aci_dir() / f"{slug}.shells.jsonl"
    
    archive_data = None
    if archive_path.exists():
        try:
            archive_data = json.loads(archive_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    
    shells_list = []
    if shells_path.exists():
        try:
            with open(shells_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        shells_list.append(json.loads(line))
        except Exception:
            pass
    
    return archive_data, shells_list


def _get_annual_capacity() -> int:
    """Get annual capacity in MU from physics constants."""
    from src.app.coordination import capacity_for_window, SECONDS_PER_YEAR
    return capacity_for_window(SECONDS_PER_YEAR)


# ---- Health Check ----
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# ---- List Programs ----
@app.get("/api/programs")
def list_programs():
    programs = []
    for f in sorted(programs_dir().glob("*.md")):
        if f.name.startswith("_"):
            continue
        slug = f.stem
        program_id = None
        id_path = aci_dir() / f"{slug}.id"
        if id_path.exists():
            program_id = id_path.read_text(encoding="utf-8").strip()
        
        # Get governance flags
        status, signed, _ = get_bundle_status(slug)
        verified = (status == "Verified")
        
        programs.append({
            "slug": slug,
            "program_id": program_id,
            "signed": signed,
            "verified": verified,
        })
    return {"programs": programs}


# ---- Create Program ----
class CreateProgramRequest(BaseModel):
    slug: str


@app.post("/api/programs")
def create_program(req: CreateProgramRequest):
    slug = req.slug.strip().lower()

    # Validate slug
    if not re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$", slug):
        raise HTTPException(
            400, "Invalid slug. Use lowercase letters, numbers, and hyphens."
        )

    program_path = programs_dir() / f"{slug}.md"
    if program_path.exists():
        raise HTTPException(400, "Program already exists.")

    # Write template
    program_path.write_text(templates.PROGRAM_TEMPLATE_MD, encoding="utf-8")

    # Generate ID
    program_id = store.ensure_program_id(slug)

    # Sync to create initial artifacts
    store.sync_program(atlas_dir(), program_path)

    return {"status": "created", "slug": slug, "program_id": program_id}


# ---- Get Program ----
@app.get("/api/programs/{slug}")
def get_program(slug: str):
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")

    # Parse editable fields from markdown
    parsed_slug, domain_counts, principle_counts, unit, notes, agents, agencies = (
        store.parse_program_from_markdown(program_path)
    )

    # Check if we have event log (real mode) vs simulation mode
    events_path = aci_dir() / f"{slug}.events.jsonl"
    has_event_log = False
    if events_path.exists():
        derived_counts = store.derive_domain_counts_from_events(events_path)
        total_derived = sum(derived_counts.values())
        if total_derived > 0:
            domain_counts = derived_counts
            has_event_log = True

    # Read computed report
    report_path = aci_dir() / f"{slug}.report.json"
    report = None
    last_synced = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mtime = report_path.stat().st_mtime
        last_synced = datetime.fromtimestamp(mtime).isoformat()

    # Get governance status
    bundle_status, signed, signer_fingerprint = get_bundle_status(slug)
    verified = (bundle_status == "Verified")

    # Get ecology data
    archive_data, shells_list = get_ecology_data(slug)
    
    ecology = None
    if archive_data:
        ecology = {
            "total_capacity_MU": archive_data.get("total_capacity_MU", 0),
            "used_capacity_MU": archive_data.get("used_capacity_MU", 0),
            "free_capacity_MU": archive_data.get("free_capacity_MU", 0),
            "shells": shells_list,
        }

    return {
        "editable": {
            "slug": parsed_slug,
            "unit": unit,
            "domain_counts": domain_counts,
            "principle_counts": principle_counts,
            "notes": notes,
            "agents": agents,
            "agencies": agencies,
        },
        "report": report,
        "last_synced": last_synced,
        "has_event_log": has_event_log,  # Flag for UI to show read-only mode
        "governance": {
            "status": bundle_status,
            "signed": signed,
            "signer_fingerprint": signer_fingerprint,
            "verified": verified,
        },
        "ecology": ecology,
    }


# ---- Update Program ----
class DomainCounts(BaseModel):
    economy: int
    employment: int
    education: int


class PrincipleCounts(BaseModel):
    GMT: int
    GTD: int
    ICV: int
    IVD: int
    IIA: int
    IAD: int
    ICI: int
    IID: int


class UpdateProgramRequest(BaseModel):
    unit: str
    domain_counts: DomainCounts
    principle_counts: PrincipleCounts
    notes: str
    agents: str
    agencies: str


@app.put("/api/programs/{slug}")
def update_program(slug: str, req: UpdateProgramRequest):
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")

    content = program_path.read_text(encoding="utf-8")

    # Update domain counts
    content = re.sub(
        r"(Economy[^:]*:\s*)\[(\d+)\]",
        rf'\g<1>[{req.domain_counts.economy}]',
        content,
    )
    content = re.sub(
        r"(Employment[^:]*:\s*)\[(\d+)\]",
        rf'\g<1>[{req.domain_counts.employment}]',
        content,
    )
    content = re.sub(
        r"(Education[^:]*:\s*)\[(\d+)\]",
        rf'\g<1>[{req.domain_counts.education}]',
        content,
    )

    # Update unit
    content = re.sub(
        r"(Unit:\s*)\[(daily|sprint)\]",
        rf"\g<1>[{req.unit}]",
        content,
        flags=re.IGNORECASE,
    )

    # Update principle counts
    for abbrev in ["GMT", "GTD", "ICV", "IVD", "IIA", "IAD", "ICI", "IID"]:
        if abbrev in ["GMT", "ICV", "IIA", "ICI"]:
            pattern = rf"({abbrev}\s+Alignment\s+Incidents:\s*)\[(\d+)\]"
        else:
            pattern = rf"({abbrev}\s+Displacement\s+Incidents:\s*)\[(\d+)\]"
        # Get value from Pydantic model using getattr
        value = getattr(req.principle_counts, abbrev)
        content = re.sub(
            pattern,
            rf"\g<1>[{value}]",
            content,
            flags=re.IGNORECASE,
        )

    # Update participants (agents) section
    agents_pattern = r'(###\s+Agents\s*\n+)(.*?)(?=###|^##|\Z)'
    agents_content = req.agents if req.agents.strip() else "(Names of people involved in this program)"
    if re.search(agents_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE):
        content = re.sub(
            agents_pattern,
            rf'\g<1>{agents_content}\n\n',
            content,
            flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
    
    # Update participants (agencies) section
    agencies_pattern = r'(###\s+Agencies\s*\n+)(.*?)(?=^##|\Z)'
    agencies_content = req.agencies if req.agencies.strip() else "(Names of agencies involved in this program)"
    if re.search(agencies_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE):
        content = re.sub(
            agencies_pattern,
            rf'\g<1>{agencies_content}\n\n',
            content,
            flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )

    # Update notes section
    # Template format: ## NOTES on one line, --- on the next line
    notes_pattern = r'(^##\s+NOTES\s*\n---?\s*\n)(.*?)(?=^##|\Z)'
    
    if re.search(notes_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE):
        # Replace existing NOTES section
        def replace_notes(match):
            if req.notes.strip():
                return match.group(1) + req.notes + "\n"
            else:
                return match.group(1) + "(Add context or key observations for this program)\n"
        content = re.sub(
            notes_pattern,
            replace_notes,
            content,
            flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
    else:
        # Append NOTES section at the end
        notes_content = req.notes if req.notes.strip() else "(Add context or key observations for this program)"
        notes_section = f"\n\n## NOTES\n---\n\n{notes_content}\n"
        content = content.rstrip() + notes_section

    # Write updated content
    program_path.write_text(content, encoding="utf-8")

    # Sync immediately
    store.sync_program(atlas_dir(), program_path)

    # Return updated state
    return get_program(slug)


# ---- Sync Program ----
@app.post("/api/programs/{slug}/sync")
def sync_program_endpoint(slug: str):
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")
    
    # Sync the program
    store.sync_program(atlas_dir(), program_path)
    
    # Sign bundle if enabled
    if get_sign_bundle_on_sync():
        from cryptography.hazmat.primitives import serialization
        
        # Try to load private key from config
        private_key = None
        config_path = aci_dir() / ".config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if "signing_key_pem" in config:
                    key_bytes = config["signing_key_pem"].encode("utf-8")
                    private_key = serialization.load_pem_private_key(
                        key_bytes, password=None
                    )
            except Exception:
                pass
        
        # Bundle will be signed if key is available
        try:
            store.bundle_program(atlas_dir(), program_path, private_key)
        except Exception:
            pass
    
    # Return updated state
    return get_program(slug)


# ---- Delete Program ----
@app.delete("/api/programs/{slug}")
def delete_program(slug: str):
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")

    # Remove markdown
    program_path.unlink()

    # Remove artifacts
    for ext in [
        ".bytes", ".events.jsonl", ".report.json", ".report.md", ".id",
        ".grants.jsonl", ".shells.jsonl", ".archive.json",
    ]:
        artifact = aci_dir() / f"{slug}{ext}"
        if artifact.exists():
            artifact.unlink()

    # Remove bundle
    bundle = bundles_dir() / f"{slug}.zip"
    if bundle.exists():
        bundle.unlink()

    return {"status": "deleted"}


# ---- Download Bundle ----
@app.get("/api/programs/{slug}/bundle")
def download_bundle(slug: str):
    # Ensure bundle exists by creating it
    program_path = programs_dir() / f"{slug}.md"
    if not program_path.exists():
        raise HTTPException(404, "Program not found.")

    # Create/update bundle (with signing if enabled)
    private_key = None
    if get_sign_bundle_on_sync():
        from cryptography.hazmat.primitives import serialization
        
        config_path = aci_dir() / ".config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if "signing_key_pem" in config:
                    key_bytes = config["signing_key_pem"].encode("utf-8")
                    private_key = serialization.load_pem_private_key(
                        key_bytes, password=None
                    )
            except Exception:
                pass
    
    try:
        store.bundle_program(atlas_dir(), program_path, private_key)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    bundle_path = bundles_dir() / f"{slug}.zip"
    if not bundle_path.exists():
        raise HTTPException(404, "Bundle not found. Sync the program first.")
    return FileResponse(
        bundle_path, filename=f"{slug}.zip", media_type="application/zip"
    )


# ---- Verify Bundle ----
@app.post("/api/programs/{slug}/verify")
def verify_bundle_endpoint(slug: str):
    """Verify the latest bundle for a program."""
    bundle_path = bundles_dir() / f"{slug}.zip"
    if not bundle_path.exists():
        raise HTTPException(404, "Bundle not found. Export a bundle first.")
    
    try:
        verified = store.verify_bundle(atlas_dir(), bundle_path)
        if verified:
            return {"status": "Verified", "verified": True}
        else:
            return {"status": "Verification failed", "verified": False}
    except Exception as e:
        raise HTTPException(500, f"Verification error: {str(e)}")


# ---- Get/Set Sign Bundle on Sync ----
@app.get("/api/config/sign-bundle-on-sync")
def get_sign_bundle_on_sync_endpoint():
    """Get 'sign bundle on sync' setting and key status."""
    return {
        "sign_bundle_on_sync": get_sign_bundle_on_sync(),
        "has_signing_key": _has_private_key()
    }


class SignBundleOnSyncRequest(BaseModel):
    sign_bundle_on_sync: bool


@app.put("/api/config/sign-bundle-on-sync")
def set_sign_bundle_on_sync_endpoint(req: SignBundleOnSyncRequest):
    """Set 'sign bundle on sync' setting."""
    set_sign_bundle_on_sync(req.sign_bundle_on_sync)
    return {
        "sign_bundle_on_sync": get_sign_bundle_on_sync(),
        "has_signing_key": _has_private_key()
    }


# ---- Get Annual Capacity ----
@app.get("/api/capacity/annual")
def get_annual_capacity_endpoint():
    """Get annual capacity in MU from physics constants."""
    return {"annual_capacity_MU": _get_annual_capacity()}


# ---- Glossary ----
@app.get("/api/glossary")
def get_glossary():
    return {
        "A_STAR": A_STAR,
        "domains": {
            "economy": {
                "name": "Economy",
                "description": "The domain of CGM operations and systemic resource flows.",
            },
            "employment": {
                "name": "Employment",
                "description": "The domain of Gyroscope work categories and human contribution patterns.",
            },
            "education": {
                "name": "Education",
                "description": "The domain of THM capacities and epistemic development.",
            },
        },
        "alignment": {
            "GMT": {
                "name": "Governance Management Traceability",
                "description": "The capacity to understand and maintain the chain of authority from human sources to outputs.",
            },
            "ICV": {
                "name": "Information Curation Variety",
                "description": "The capacity to recognise and preserve diversity in information sources.",
            },
            "IIA": {
                "name": "Inference Interaction Accountability",
                "description": "The capacity to accept responsibility for conclusions and decisions.",
            },
            "ICI": {
                "name": "Intelligence Cooperation Integrity",
                "description": "The capacity to maintain coherent reasoning over time and context.",
            },
        },
        "displacement": {
            "GTD": {
                "name": "Governance Traceability Displacement",
                "risk": "Approaching Derivative Authority and Agency as Original.",
                "description": "Occurs when a derivative system is treated as if it were an autonomous original source.",
            },
            "IVD": {
                "name": "Information Variety Displacement",
                "risk": "Approaching Derivative Authority without Agency as Original.",
                "description": "Occurs when derivative authority is treated as original authority.",
            },
            "IAD": {
                "name": "Inference Accountability Displacement",
                "risk": "Approaching Derivative Agency without Authority as Original.",
                "description": "Occurs when derivative agency is treated as original agency.",
            },
            "IID": {
                "name": "Intelligence Integrity Displacement",
                "risk": "Approaching Original Authority and Agency as Derivative.",
                "description": "Occurs when original authority and agency are devalued as inferior to derivative processing.",
            },
        },
    }


# ---- Serve Static Files (for production) ----
# Check if dist folder exists
ui_dist = Path(__file__).parent.parent / "ui" / "dist"
if ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="static")

