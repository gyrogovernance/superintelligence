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
import json
import re

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
def projects_dir() -> Path:
    return store.get_projects_dir()


def aci_dir() -> Path:
    return store.get_aci_dir()


def bundles_dir() -> Path:
    return store.get_bundles_dir()


def atlas_dir() -> Path:
    return store.get_atlas_dir()


# ---- List Projects ----
@app.get("/api/projects")
def list_projects():
    projects = []
    for f in sorted(projects_dir().glob("*.md")):
        if f.name.startswith("_"):
            continue
        slug = f.stem
        project_id = None
        id_path = aci_dir() / f"{slug}.id"
        if id_path.exists():
            project_id = id_path.read_text(encoding="utf-8").strip()
        projects.append({"slug": slug, "project_id": project_id})
    return {"projects": projects}


# ---- Create Project ----
class CreateProjectRequest(BaseModel):
    slug: str


@app.post("/api/projects")
def create_project(req: CreateProjectRequest):
    slug = req.slug.strip().lower()

    # Validate slug
    if not re.match(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$|^[a-z0-9]$", slug):
        raise HTTPException(
            400, "Invalid slug. Use lowercase letters, numbers, and hyphens."
        )

    project_path = projects_dir() / f"{slug}.md"
    if project_path.exists():
        raise HTTPException(400, "Project already exists.")

    # Write template
    project_path.write_text(templates.PROJECT_TEMPLATE_MD, encoding="utf-8")

    # Generate ID
    project_id = store.ensure_project_id(slug)

    # Sync to create initial artifacts
    store.sync_project(atlas_dir(), project_path)

    return {"status": "created", "slug": slug, "project_id": project_id}


# ---- Get Project ----
@app.get("/api/projects/{slug}")
def get_project(slug: str):
    project_path = projects_dir() / f"{slug}.md"
    if not project_path.exists():
        raise HTTPException(404, "Project not found.")

    # Parse editable fields from markdown
    parsed_slug, domain_counts, principle_counts, unit, notes = (
        store.parse_project_from_markdown(project_path)
    )

    # Read computed report
    report_path = aci_dir() / f"{slug}.report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = None

    return {
        "editable": {
            "slug": parsed_slug,
            "unit": unit,
            "domain_counts": domain_counts,
            "principle_counts": principle_counts,
            "notes": notes,
        },
        "report": report,
    }


# ---- Update Project ----
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


class UpdateProjectRequest(BaseModel):
    unit: str
    domain_counts: DomainCounts
    principle_counts: PrincipleCounts
    notes: str


@app.put("/api/projects/{slug}")
def update_project(slug: str, req: UpdateProjectRequest):
    project_path = projects_dir() / f"{slug}.md"
    if not project_path.exists():
        raise HTTPException(404, "Project not found.")

    content = project_path.read_text(encoding="utf-8")

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

    # Update notes section
    notes_pattern = r'(^##\s+NOTES\s*---?\s*\n)(.*?)(?=^##|\Z)'
    
    if re.search(notes_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE):
        # Replace existing NOTES section
        def replace_notes(match):
            if req.notes.strip():
                return match.group(1) + req.notes + "\n"
            else:
                return match.group(1) + "(Add context or key observations for this project)\n"
        content = re.sub(
            notes_pattern,
            replace_notes,
            content,
            flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
    else:
        # Append NOTES section at the end
        notes_content = req.notes if req.notes.strip() else "(Add context or key observations for this project)"
        notes_section = f"\n\n## NOTES\n---\n\n{notes_content}\n"
        content = content.rstrip() + notes_section

    # Write updated content
    project_path.write_text(content, encoding="utf-8")

    # Sync immediately
    store.sync_project(atlas_dir(), project_path)

    # Return updated state
    return get_project(slug)


# ---- Delete Project ----
@app.delete("/api/projects/{slug}")
def delete_project(slug: str):
    project_path = projects_dir() / f"{slug}.md"
    if not project_path.exists():
        raise HTTPException(404, "Project not found.")

    # Remove markdown
    project_path.unlink()

    # Remove artifacts
    for ext in [".bytes", ".events.jsonl", ".report.json", ".report.md", ".id"]:
        artifact = aci_dir() / f"{slug}{ext}"
        if artifact.exists():
            artifact.unlink()

    # Remove bundle
    bundle = bundles_dir() / f"{slug}.zip"
    if bundle.exists():
        bundle.unlink()

    return {"status": "deleted"}


# ---- Download Bundle ----
@app.get("/api/projects/{slug}/bundle")
def download_bundle(slug: str):
    # Ensure bundle exists by creating it
    project_path = projects_dir() / f"{slug}.md"
    if not project_path.exists():
        raise HTTPException(404, "Project not found.")

    # Create/update bundle
    try:
        store.bundle_project(atlas_dir(), project_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    bundle_path = bundles_dir() / f"{slug}.zip"
    if not bundle_path.exists():
        raise HTTPException(404, "Bundle not found. Sync the project first.")
    return FileResponse(
        bundle_path, filename=f"{slug}.zip", media_type="application/zip"
    )


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

