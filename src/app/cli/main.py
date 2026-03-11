"""
AIR CLI main entry point.
Default behavior: sync all programs, verify, and print status.
"""

import sys
from types import SimpleNamespace

from . import store, ui


def cmd_sync_and_verify_all(args):
    """Default behavior: sync all programs and verify artifacts."""
    print(ui.header("System Status"))

    programs_dir = store.get_programs_dir()
    program_files = sorted([f for f in programs_dir.glob("*.md") if not f.name.startswith("_")])

    if not program_files:
        print()
        ui.success("System Synced and Verified")
        print()
        ui.warn("No program contracts found in data/programs/")
        print()
        print("To create a program:")
        print("  1. Copy the template: cp data/programs/_template.md data/programs/my-program.md")
        print("  2. Edit my-program.md and fill in the domain counts and incident counts")
        print("  3. Run this command again to sync and verify")
        return True

    synced_count = 0
    failed_count = 0
    warnings = []

    print(f"\nSyncing {len(program_files)} program(s)...")

    for program_file in program_files:
        program_slug, _, _, _, _, _, _ = store.parse_program_from_markdown(program_file)

        try:
            store.sync_program(program_file)
            try:
                bundle_path = store.bundle_program(program_file)
            except Exception as e:
                warnings.append(f"Failed to bundle {program_slug}: {e}")
            synced_count += 1
        except Exception as e:
            failed_count += 1
            warnings.append(f"Failed to sync {program_slug}: {e}")

    verified_count = 0
    bundle_verified_count = 0
    for program_file in program_files:
        program_slug, _, _, _, _, _, _ = store.parse_program_from_markdown(program_file)

        try:
            aci_dir = store.get_aci_dir()
            bytes_path = aci_dir / f"{program_slug}.bytes"
            events_path = aci_dir / f"{program_slug}.events.jsonl"

            if bytes_path.exists() or events_path.exists():
                store.replay_program(program_slug)
                verified_count += 1
        except Exception as e:
            warnings.append(f"Failed to verify {program_slug}: {e}")

        bundles_dir = store.get_bundles_dir()
        bundle_path = bundles_dir / f"{program_slug}.zip"
        if bundle_path.exists():
            try:
                if store.verify_bundle(bundle_path):
                    bundle_verified_count += 1
                else:
                    warnings.append(f"Bundle verification failed for {program_slug}")
            except Exception as e:
                warnings.append(f"Failed to verify bundle for {program_slug}: {e}")

    print()
    print(ui.header("Summary"))
    print(ui.kv("Programs synced", str(synced_count)))
    print(ui.kv("Programs verified", str(verified_count)))
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
    store.ensure_workspace()
    store.ensure_templates()

    try:
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