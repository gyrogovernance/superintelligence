#!/usr/bin/env python3
"""
Test script for one-article ingestion and continuation.

This script:
1. Reads a single Wikipedia article from toys/training/wiki_test.txt
2. Creates a fresh GyroSI agent with private knowledge store
3. Ingests the article (learns from it)
4. Seeds with a short prefix from the article
5. Generates continuation and decodes to text
6. Compares with expected continuation from the article

This tests whether the system can learn from one article and continue from a seed.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baby.information import encode_text_with_sep, decode_text
from baby.intelligence import GyroSI
from baby.contracts import AgentConfig


def build_agent(private_knowledge_path: Path) -> GyroSI:
    """
    Create a fresh GyroSI agent for testing.

    Args:
        private_knowledge_path: Path to private knowledge store

    Returns:
        GyroSI: Configured agent instance
    """
    # Create dummy public knowledge file
    dummy_public = PROJECT_ROOT / "toys/training/dummy_public_knowledge.bin"
    if not dummy_public.exists():
        dummy_public.parent.mkdir(parents=True, exist_ok=True)
        dummy_public.write_bytes(b"")

    config: AgentConfig = {
        "ontology_path": str(PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"),
        "phenomenology_map_path": str(PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"),
        "epistemology_path": str(PROJECT_ROOT / "memories/public/meta/epistemology.npy"),
        "public_knowledge_path": str(dummy_public),
        "private_knowledge_path": str(private_knowledge_path),
        "learn_batch_size": 100,
        "enable_phenomenology_storage": True,
        "preferences": {},
    }

    return GyroSI(config, agent_id="test_continuation", base_path=PROJECT_ROOT)


def find_continuation_in_text(text: str, seed: str, max_chars: int = 200) -> Optional[str]:
    """
    Find the continuation of seed in the original text.

    Args:
        text: Full article text
        seed: Seed text to find
        max_chars: Maximum characters to return as continuation

    Returns:
        Optional[str]: Continuation text if found, None otherwise
    """
    try:
        start_idx = text.index(seed)
        continuation_start = start_idx + len(seed)
        continuation_end = min(continuation_start + max_chars, len(text))
        return text[continuation_start:continuation_end].strip()
    except ValueError:
        return None


def main() -> int:
    """
    Main test function.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Configuration
    article_path = PROJECT_ROOT / "toys/training/wiki_test.txt"
    knowledge_path = PROJECT_ROOT / "toys/training/knowledge/test_continuation.bin"
    max_new_tokens = 48

    # Seed text from the article
    seed = "Algorithm\n\nIn mathematics and computer science, an algorithm () is"

    print("🧪 Testing one-article ingestion and continuation")
    print(f"   Article: {article_path}")
    print(f"   Knowledge store: {knowledge_path}")
    print(f"   Seed: '{seed[:50]}...'")
    print(f"   Max tokens: {max_new_tokens}")
    print("-" * 60)

    # Check if article exists
    if not article_path.exists():
        print(f"❌ Error: Article file not found: {article_path}")
        return 1

    # Read the article
    try:
        article_text = article_path.read_text(encoding="utf-8")
        print(f"📖 Read article: {len(article_text)} characters")
    except Exception as e:
        print(f"❌ Error reading article: {e}")
        return 1

    # Find expected continuation in the original text
    expected_continuation = find_continuation_in_text(article_text, seed)
    if expected_continuation:
        print(f"📋 Expected continuation: '{expected_continuation[:50]}...'")
    else:
        print("⚠️  Warning: Could not find seed in article text")

    # Create knowledge directory
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)

    # Build fresh agent
    print("🤖 Creating fresh agent...")
    agent = build_agent(knowledge_path)

    try:
        # Encode article to masked introns
        print("🔧 Encoding article to masked introns...")
        intron_bytes = encode_text_with_sep(article_text)
        print(f"   Encoded: {len(intron_bytes)} bytes")

        # Ingest article (learn from it)
        print("📚 Ingesting article...")
        agent.ingest_bulk(intron_bytes)
        agent._commit_if_needed()

        # Check knowledge store size
        if knowledge_path.exists():
            knowledge_size = knowledge_path.stat().st_size
            print(f"   Knowledge store size: {knowledge_size} bytes")
        else:
            print("   Knowledge store not created")

        # Generate continuation from seed
        print("🎯 Generating continuation...")
        seed_bytes = seed.encode("utf-8")

        # Use respond method to generate continuation
        response_bytes = agent.respond(seed_bytes, max_new_tokens=max_new_tokens)

        # Decode response to text
        try:
            generated_text = decode_text(response_bytes)
            print(f"   Generated: {len(generated_text)} characters")
        except Exception as e:
            print(f"   Decode error: {e}")
            generated_text = response_bytes.decode("utf-8", errors="ignore")

        # Print results
        print("\n" + "=" * 60)
        print("📋 RESULTS")
        print("=" * 60)
        print(f"Seed: '{seed}'")
        print(f"Generated continuation: '{generated_text}'")

        if expected_continuation:
            print(f"Expected continuation: '{expected_continuation}'")

            # Simple similarity check
            if expected_continuation.lower() in generated_text.lower():
                print("✅ SUCCESS: Generated text contains expected continuation!")
            else:
                print("❌ FAILURE: Generated text does not match expected continuation")
        else:
            print("⚠️  No expected continuation available for comparison")

        # Save results to JSON
        results = {
            "seed": seed,
            "generated_continuation": generated_text,
            "expected_continuation": expected_continuation,
            "max_new_tokens": max_new_tokens,
            "knowledge_size": knowledge_path.stat().st_size if knowledge_path.exists() else 0,
        }

        results_path = knowledge_path.with_suffix(".results.json")
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {results_path}")

        return 0

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Clean up
        try:
            agent.close()
            print("🧹 Agent closed successfully")
        except Exception as e:
            print(f"⚠️  Error closing agent: {e}")


if __name__ == "__main__":
    sys.exit(main())
