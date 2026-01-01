"""
AIR CLI main entry point.
Default behavior: sync all projects, verify, and print status.
"""

import sys
from types import SimpleNamespace

from . import store, ui


def cmd_sync_and_verify_all(args):
    """Default behavior: sync all projects and verify artifacts."""
    from src.router.atlas import build_all
    
    # Print ASCII-safe banner
    print("\nAIR")
    print("Alignment Infrastructure Routing")
    print("-----------------------------------")
    print(ui.header("System Status"))
    
    # Ensure atlas exists (build if missing)
    atlas_dir = store.get_atlas_dir()
    required_files = ["ontology.npy", "epistemology.npy", "phenomenology.npz"]
    missing = [f for f in required_files if not (atlas_dir / f).exists()]
    if missing:
        print("Building atlas...")
        build_all(atlas_dir)
        # Re-check after build
        still_missing = [f for f in required_files if not (atlas_dir / f).exists()]
        if still_missing:
            ui.error(f"Atlas build failed: missing {still_missing}")
            return False
        ui.success("Atlas built")
    
    # Sync all projects
    projects_dir = store.get_projects_dir()
    project_files = sorted([f for f in projects_dir.glob("*.md") if f.name != "project_template.md"])
    
    if not project_files:
        print()
        ui.success("System Synced and Verified")
        return True
    
    synced_count = 0
    failed_count = 0
    warnings = []
    
    print(f"\nSyncing {len(project_files)} project(s)...")
    
    for project_file in project_files:
        project_meta, _ = store.parse_frontmatter(project_file)
        project_slug = project_meta.get("project_slug", project_file.stem)
        
        try:
            store.sync_project(atlas_dir, project_file)
            # Create bundle after successful sync
            try:
                bundle_path = store.bundle_project(atlas_dir, project_file)
            except Exception as e:
                warnings.append(f"Failed to bundle {project_slug}: {e}")
            synced_count += 1
        except Exception as e:
            failed_count += 1
            warnings.append(f"Failed to sync {project_slug}: {e}")
    
    # Verify all projects (replay artifacts and verify bundles)
    verified_count = 0
    bundle_verified_count = 0
    for project_file in project_files:
        project_meta, _ = store.parse_frontmatter(project_file)
        project_slug = project_meta.get("project_slug", project_file.stem)
        
        try:
            # Try to replay project artifacts
            aci_dir = store.get_aci_dir()
            bytes_path = aci_dir / f"{project_slug}.bytes"
            events_path = aci_dir / f"{project_slug}.events.jsonl"
            
            if bytes_path.exists() or events_path.exists():
                store.replay_project(atlas_dir, project_slug)
                verified_count += 1
        except Exception as e:
            warnings.append(f"Failed to verify {project_slug}: {e}")
        
        # Verify bundle
        bundles_dir = store.get_bundles_dir()
        bundle_path = bundles_dir / f"{project_slug}.zip"
        if bundle_path.exists():
            try:
                if store.verify_bundle(atlas_dir, bundle_path):
                    bundle_verified_count += 1
                else:
                    warnings.append(f"Bundle verification failed for {project_slug}")
            except Exception as e:
                warnings.append(f"Failed to verify bundle for {project_slug}: {e}")
    
    # Print status summary
    print()
    print(ui.header("Summary"))
    print(ui.kv("Projects synced", str(synced_count)))
    print(ui.kv("Projects verified", str(verified_count)))
    print(ui.kv("Bundles verified", str(bundle_verified_count)))
    if failed_count > 0:
        print(ui.kv("Failures", str(failed_count)))
    
    print()
    if failed_count == 0 and len(warnings) == 0:
        ui.success("System Synced and Verified")
        return True
    else:
        ui.warn("System Synced and Verified (with warnings)")
        for warning in warnings:
            ui.warn(f"  {warning}")
        if len(warnings) > 0 or failed_count > 0:
            print()
            ui.warn("Exit code is non-zero because warnings or failures were detected.")
        return False


def main():
    """Main entry point - always runs sync and verify."""
    # Ensure workspace exists and templates are installed
    store.ensure_workspace()
    store.ensure_templates()
    
    try:
        # Always run default behavior (ignore argv)
        args = SimpleNamespace()
        success = cmd_sync_and_verify_all(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nGoodbye.")
        sys.exit(0)
    except Exception as e:
        ui.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
