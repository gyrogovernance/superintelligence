#!/usr/bin/env python3
"""Recreate empty memory files for fresh testing."""

from pathlib import Path


def recreate_memory_files():
    """Recreate the necessary memory files with proper empty content."""

    memory_dir = Path("memories/public/knowledge")

    # Create directory if it doesn't exist
    memory_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create empty address_memory.dat
    address_memory_path = memory_dir / "address_memory.dat"
    with open(address_memory_path, "wb") as f:
        # Write empty content - the system will initialize it properly
        f.write(b"")

    print(f"✅ Created empty address_memory.dat")

    # 2. Create basic address_memory.json metadata
    address_metadata_path = memory_dir / "address_memory.json"
    import json

    metadata = {
        "version": "1.0",
        "total_entries": 0,
        "created": "2024-01-01T00:00:00Z",
        "last_modified": "2024-01-01T00:00:00Z",
    }

    with open(address_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Created address_memory.json")

    # 3. Create empty passive_memory.bin
    passive_memory_path = memory_dir / "passive_memory.bin"
    with open(passive_memory_path, "wb") as f:
        # Write empty content - the system will initialize it properly
        f.write(b"")

    print(f"✅ Created empty passive_memory.bin")

    print(f"\n🎉 Memory files recreated successfully!")
    print(f"📁 Location: {memory_dir.absolute()}")
    print(f"\nYou can now run the knowledge test with fresh memory.")


if __name__ == "__main__":
    recreate_memory_files()
