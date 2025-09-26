#!/usr/bin/env python3
"""
GyroSI Wikipedia Training Pipeline - Token-Aware Stream Compiler

This script converts Wikipedia text dumps into compact "gyro-tapes" (raw intron streams)
and optionally updates token-aware knowledge stores. It's optimized for the 0.9.6.7
token-aware refactored architecture with high-performance batch processing.

Key Features:
- Streams articles one-by-one to avoid memory issues
- Uses optimized one-shot encoding for maximum throughput
- Converts text directly to masked introns via gyro_encode
- Writes compact .gyro tape files (~1.5 bytes/token)
- Optionally updates token-aware knowledge store with batch processing
- Real-time progress tracking with performance metrics
- PEP8 compliant with proper typing for static analysis tools

Performance Optimizations:
- One-shot encoding instead of per-token loops (2-3x faster)
- Batch writes with 5000-item buffers (5x fewer disk flushes)
- Single ingest calls instead of per-byte processing
- Optimized for 1000+ tokens/second on modern hardware

Usage:
    # Simple Wikipedia to tape only (fastest) - uses default knowledge directory
    python toys/training/wikipedia_eng.py --simple

    # Simple Wikipedia to tape with learning - uses default knowledge directory
    python toys/training/wikipedia_eng.py --simple --learn

    # Simple Wikipedia to tape with learning - uses default knowledge directory
    python toys/training/wikipedia_eng.py --replay toys/training/knowledge/wikipedia_simple.gyro --learn

    # Full Wikipedia to tape with learning - uses default knowledge directory
    python toys/training/wikipedia_eng.py --full --learn

    # Limit articles for testing - uses default knowledge directory
    python toys/training/wikipedia_eng.py --simple --limit 1000

    # Specify custom output location
    python toys/training/wikipedia_eng.py --simple -o custom_output.gyro
"""

import argparse
import gzip
import json
import sys
import time
import warnings
from itertools import islice
from pathlib import Path
from typing import Iterator, Iterable, Optional, List, Union, Dict, cast, Any
import numpy as np

# Suppress all warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baby.information import encode_text_with_sep as gyro_encode  # noqa: E402
from baby.intelligence import GyroSI  # noqa: E402
from baby.contracts import AgentConfig, PreferencesConfig  # noqa: E402
from baby.policies import prune_and_compact_store  # noqa: E402

# Default constants
DEFAULT_LOG_INTERVAL = 50_000  # Default logging interval
DEFAULT_BLANK_LINES = 3  # Default number of blank lines that separate articles


def iter_wiki_articles(files: Iterable[Path], blank_line_threshold: int = DEFAULT_BLANK_LINES) -> Iterator[str]:
    """
    Yield one article (str) at a time from Wikipedia dataset files.

    Articles are separated by blank_line_threshold or more blank lines.

    Args:
        files: Iterable of file paths to process
        blank_line_threshold: Number of consecutive blank lines to define article boundary

    Yields:
        str: Complete article text as a single string
    """
    for file_path in files:
        # Handle both .txt and .gz files
        open_func = gzip.open if file_path.suffix == ".gz" else open

        with open_func(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            buffer: List[str] = []
            blank_line_count = 0

            for line in f:
                if line.strip():
                    buffer.append(line)
                    blank_line_count = 0
                else:
                    blank_line_count += 1
                    # Configurable blank line threshold for article boundary
                    if blank_line_count >= blank_line_threshold and buffer:
                        yield "".join(buffer)
                        buffer.clear()

            # Don't forget the last article if file doesn't end with blank lines
            if buffer:
                yield "".join(buffer)


def build_agent(private_knowledge_path: Path) -> GyroSI:
    """
    Create a private GyroSI agent for training.

    Args:
        private_knowledge_path: Path to private knowledge store file

    Returns:
        GyroSI: Configured agent instance
    """
    # Create dummy public knowledge file if it doesn't exist
    dummy_public = PROJECT_ROOT / "toys/training/dummy_public_knowledge.bin"
    if not dummy_public.exists():
        dummy_public.parent.mkdir(parents=True, exist_ok=True)
        dummy_public.write_bytes(b"")

    # Load preferences for auto-pruning
    prefs_path = PROJECT_ROOT / "memories/memory_preferences.json"
    if prefs_path.exists():
        with open(prefs_path) as f:
            preferences = json.load(f)
    else:
        # Default preferences without legacy pruning settings
        preferences = {}

    # Configure agent preferences (keep pruning nested)
    preferences_config = cast(PreferencesConfig, preferences)

    config: AgentConfig = {
        "ontology_path": str(PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"),
        "phenomenology_map_path": str(PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"),
        "epistemology_path": str(PROJECT_ROOT / "memories/public/meta/epistemology.npy"),
        "public_knowledge_path": str(dummy_public),
        "private_knowledge_path": str(private_knowledge_path),
        "learn_batch_size": 5000,  # write threshold for PhenotypeStore (optimized for bulk replay)
        "enable_phenomenology_storage": True,  # <-- top-level (so CanonicalView is enabled explicitly)
        "preferences": preferences_config,
    }

    return GyroSI(config, agent_id="wiki_trainer", base_path=PROJECT_ROOT)


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def format_time(seconds: float) -> str:
    """
    Format seconds into human readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string (HH:MM:SS or MM:SS)
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def compile_stream(
    articles: Iterable[str],
    output_tape_path: Path,
    agent: Optional[GyroSI] = None,
    limit: Optional[int] = None,
    log_interval: int = DEFAULT_LOG_INTERVAL,
) -> Dict[str, Union[int, float, str]]:
    """
    Compile Wikipedia articles into a gyro-tape and optionally update knowledge store.

    Args:
        articles: Iterable of article texts
        output_tape_path: Path to output .gyro file
        agent: Optional GyroSI agent for learning (via process_egress)
        limit: Optional limit on number of articles to process
        log_interval: How often to log progress (in number of articles)

    Returns:
        Dictionary of statistics about the compilation
    """
    # We'll still need a tokenizer to count tokens for the stats,
    # but only once per article (fast).
    # from tokenizers import Tokenizer
    # hf_tok = Tokenizer.from_pretrained("bert-base-uncased")

    # Create output directory
    output_tape_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize counters
    articles_processed = 0
    tokens_processed = 0
    bytes_written = 0
    start_time = time.time()
    last_log_time = start_time

    print("🚀 Starting compilation...")
    print(f"   Output: {output_tape_path}")
    print(f"   Learning: {'Yes' if agent else 'No'}")
    if limit:
        print(f"   Limit: {limit:,} articles")
    print("-" * 60)

    # Open output file for writing
    with output_tape_path.open("wb") as tape_file:
        # Apply limit if specified
        article_iterator = islice(articles, limit) if limit else articles

        try:
            for article_text in article_iterator:
                articles_processed += 1

                # One-shot encode → masked introns
                intron_bytes = gyro_encode(article_text)

                # Count tokens for statistics (always enabled for progress tracking)
                # Count tokens by LEB128 terminal-byte rule: last byte has top bit 0 after unmask
                arr = np.frombuffer(intron_bytes, dtype=np.uint8)
                tokens_processed += int(np.count_nonzero(((arr ^ 0xAA) & 0x80) == 0))

                tape_file.write(intron_bytes)
                bytes_written += len(intron_bytes)

                # One ingest call (internal loop stays, but far fewer Python calls)
                if agent:
                    agent.ingest_bulk(intron_bytes)

                # Log progress periodically based on articles or time
                current_time = time.time()
                if (
                    articles_processed % log_interval == 0 or current_time - last_log_time >= 60
                ):  # At least every minute

                    elapsed = current_time - start_time
                    rate_articles = articles_processed / elapsed if elapsed > 0 else 0
                    rate_tokens = tokens_processed / elapsed if elapsed > 0 else 0
                    rate_bytes = bytes_written / elapsed / (1024 * 1024) if elapsed > 0 else 0  # MB/s

                    # Calculate ETA if we know the limit
                    eta_str = ""
                    if limit:
                        articles_remaining = limit - articles_processed
                        if rate_articles > 0:
                            eta_seconds = articles_remaining / rate_articles
                            eta_str = f" | ETA: {format_time(eta_seconds)}"

                    progress_msg = (
                        f"📊 Progress: {articles_processed:,} articles"
                        f"{f'/{limit:,}' if limit else ''} | "
                        f"{tokens_processed:,} tokens | "
                        f"{format_size(bytes_written)} written | "
                        f"{rate_articles:.0f} arts/s | "
                        f"{rate_tokens:.0f} tokens/s | "
                        f"{rate_bytes:.1f} MB/s{eta_str}"
                    )
                    print(progress_msg)
                    last_log_time = current_time

                    # Flush output to show progress in logs
                    sys.stdout.flush()

                    # Periodic commits for crash resiliency (optional)
                    if agent and (articles_processed % 5000 == 0):
                        store = agent.engine.operator.store
                        if hasattr(store, "commit"):
                            store.commit()

        except KeyboardInterrupt:
            print("\n⚠️  Compilation interrupted! Saving progress...")

        finally:
            # Always flush the tape file before closing
            tape_file.flush()

    # Commit any pending changes to knowledge store
    if agent and hasattr(agent.engine.operator.store, "commit"):
        agent.engine.operator.store.commit()
    # Removed Bloom filter saving as it's no longer used in the current architecture

    # Final statistics
    total_elapsed = time.time() - start_time
    final_rate_articles = articles_processed / total_elapsed if total_elapsed > 0 else 0
    final_rate_tokens = tokens_processed / total_elapsed if total_elapsed > 0 else 0
    final_rate_bytes = bytes_written / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0

    # Prepare stats dictionary
    stats: Dict[str, Union[int, float, str]] = {
        "articles_processed": articles_processed,
        "tokens_processed": tokens_processed,
        "bytes_written": bytes_written,
        "processing_time": total_elapsed,
        "rate_articles": final_rate_articles,
        "rate_tokens": final_rate_tokens,
        "rate_bytes": final_rate_bytes,
    }

    # Add knowledge stats if learning was enabled
    if agent:
        knowledge_path_str = agent.config.get("private_knowledge_path")
        if knowledge_path_str is not None:
            knowledge_path = Path(knowledge_path_str)
            if knowledge_path.exists():
                knowledge_size = knowledge_path.stat().st_size
                stats["knowledge_size"] = knowledge_size

    print("-" * 60)
    print("✅ Compilation completed!")
    print(f"   Articles processed: {articles_processed:,}")
    print(f"   Tokens processed: {tokens_processed:,}")
    print(f"   Tape size: {format_size(bytes_written)}")
    print(f"   Processing time: {format_time(total_elapsed)}")
    final_perf_msg = (
        f"   Performance: {final_rate_articles:.0f} arts/s | "
        f"{final_rate_tokens:.0f} tokens/s | "
        f"{final_rate_bytes:.1f} MB/s"
    )
    print(final_perf_msg)
    if agent and "knowledge_size" in stats:
        print(f"   Knowledge store: {format_size(int(stats['knowledge_size']))}")

    # Write stats to JSON file alongside the tape
    stats_path = output_tape_path.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        # Convert stats to JSON-serializable format
        json_stats = {k: v if not isinstance(v, Path) else str(v) for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)

    print(f"   Stats saved to: {stats_path}")

    return stats


def replay_tape(
    tape_path: Path,
    agent: GyroSI,
    log_interval: int = 1_000,  # Log every 1KB by default (very responsive)
) -> Dict[str, Union[int, float, str]]:
    """
    Replay a gyro-tape through an agent for learning.

    Args:
        tape_path: Path to the .gyro tape file
        agent: GyroSI agent to replay through
        log_interval: Bytes processed before next progress log

    Returns:
        Dictionary of statistics about the replay
    """
    print(f"🎬 Replaying tape: {tape_path}")

    # Get tape file size for progress tracking
    tape_size = tape_path.stat().st_size

    bytes_processed = 0
    next_log_at = log_interval
    start_time = time.time()
    last_log_time = start_time

    # Optimize reading with larger buffer
    # 4MB buffer for better throughput

    # Get initial state for tracking
    initial_state = agent.engine.get_state_info()["tensor_index"]

    with tape_path.open("rb") as f:
        try:
            while True:
                # Read in larger chunks for better performance (4MB instead of 1MB)
                chunk = f.read(4 * 1024 * 1024)  # 4MB chunks
                if not chunk:
                    break

                # Use vectorized bulk ingestion - same physics as live learning
                chunk_start_time = time.time()
                agent.ingest_bulk(chunk)  # masked bytes are exactly what this expects
                chunk_processing_time = time.time() - chunk_start_time

                bytes_processed += len(chunk)
                current_time = time.time()
                # if we've crossed the next_log_at threshold or 60s have passed
                if bytes_processed >= next_log_at or (current_time - last_log_time) >= 60:

                    # Calculate rate using time up to the end of the previous chunk for accuracy
                    elapsed = current_time - start_time
                    rate = bytes_processed / elapsed / (1024 * 1024) if elapsed > 0 else 0

                    # Calculate progress percentage and ETA
                    progress_pct = bytes_processed / tape_size * 100
                    eta_seconds = (
                        (tape_size - bytes_processed) / (bytes_processed / elapsed)
                        if elapsed > 0 and bytes_processed > 0
                        else 0
                    )

                    # Current state for tracking evolution
                    current_state = agent.engine.get_state_info()["tensor_index"]
                    state_delta = "same" if current_state == initial_state else "changed"

                    # Add detailed timing info
                    chunk_size_mb = len(chunk) / (1024 * 1024)
                    chunk_rate = chunk_size_mb / chunk_processing_time if chunk_processing_time > 0 else 0

                    progress_msg = (
                        f"   {progress_pct:.1f}% | "
                        f"Processed {format_size(bytes_processed)}/{format_size(tape_size)} | "
                        f"Rate: {rate:.3f} MB/s | "
                        f"Last chunk: {chunk_size_mb:.1f}MB in {chunk_processing_time:.1f}s ({chunk_rate:.3f} MB/s) | "
                        f"ETA: {format_time(eta_seconds)} | "
                        f"State: {state_delta}"
                    )
                    print(progress_msg)
                    last_log_time = current_time
                    next_log_at += log_interval

                    # Flush output to show progress in logs
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n⚠️  Replay interrupted!")

    # Commit changes at the end (not during the loop)
    if hasattr(agent.engine.operator.store, "commit"):
        agent.engine.operator.store.commit()
    # Removed Bloom filter saving reference

    # Final state check
    final_state = agent.engine.get_state_info()["tensor_index"]
    state_changed = final_state != initial_state

    total_elapsed = time.time() - start_time
    final_rate = bytes_processed / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0

    # Prepare stats dictionary
    stats: Dict[str, Union[int, float, str]] = {
        "tape_size": tape_size,
        "bytes_processed": bytes_processed,
        "processing_time": total_elapsed,
        "rate_bytes": final_rate,
        "state_changed": state_changed,
        "initial_state": initial_state,
        "final_state": final_state,
    }

    # Add knowledge stats
    knowledge_path_str = agent.config.get("private_knowledge_path")
    if knowledge_path_str is not None:
        knowledge_path = Path(knowledge_path_str)
        if knowledge_path.exists():
            knowledge_size = knowledge_path.stat().st_size
            stats["knowledge_size"] = knowledge_size

    print(f"✅ Replay completed: {format_size(bytes_processed)} in {format_time(total_elapsed)}")
    print(f"   Final rate: {final_rate:.3f} MB/s")
    print(f"   State evolution: {'Changed' if state_changed else 'Unchanged'}")
    if "knowledge_size" in stats:
        print(f"   Knowledge store: {format_size(int(stats['knowledge_size']))}")

    # Write stats to JSON file
    stats_path = tape_path.with_suffix(".replay.json")
    with stats_path.open("w") as f:  # type: ignore
        # Convert stats to JSON-serializable format
        def convert_numpy_types(obj: Any) -> Any:
            if hasattr(obj, "item"):  # NumPy scalar
                return obj.item()
            elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        json_stats = {k: convert_numpy_types(v) for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)  # type: ignore

    print(f"   Stats saved to: {stats_path}")

    return stats


def main() -> int:
    """
    Main entry point for the gyro-tape compiler.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="GyroSI Wikipedia Training Pipeline - Token-Aware Stream Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple Wikipedia to tape only (fastest) - uses default knowledge directory
  python toys/training/wikipedia_eng.py --simple

  # Full Wikipedia to tape with learning - uses default knowledge directory
  python toys/training/wikipedia_eng.py --full --learn

  # Limit articles for testing - uses default knowledge directory
  python toys/training/wikipedia_eng.py --simple --limit 1000

  # Replay existing tape for learning
  python toys/training/wikipedia_eng.py --replay tape.gyro --learn

  # Customize blank line threshold for article boundary
  python toys/training/wikipedia_eng.py --simple --blank-lines 2

  # Specify custom output location
  python toys/training/wikipedia_eng.py --simple -o custom_output.gyro
        """,
    )

    # Input source group
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--simple", action="store_true", help="Use Simple Wikipedia dataset")

    source_group.add_argument("--full", action="store_true", help="Use Full English Wikipedia dataset")

    source_group.add_argument("--replay", type=str, help="Replay existing .gyro tape file")

    # Output specification
    parser.add_argument(
        "-o", "--output", help="Output .gyro tape file path (optional, defaults to toys/training/knowledge/)"
    )

    # Learning option
    parser.add_argument("--learn", action="store_true", help="Also update private knowledge store")
    parser.add_argument("--compact", action="store_true", help="Compact the knowledge store at the end")

    # Processing options
    parser.add_argument("--limit", type=int, help="Stop after processing N articles (useful for testing)")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help=f"Log progress every N articles (default: {DEFAULT_LOG_INTERVAL}). "
        f"For replay: every N bytes (default: 10KB)",
    )

    parser.add_argument(
        "--blank-lines",
        type=int,
        default=DEFAULT_BLANK_LINES,
        help=f"Number of blank lines that separate articles (default: {DEFAULT_BLANK_LINES})",
    )

    args = parser.parse_args()

    # Handle replay mode
    if args.replay:
        if not args.learn:
            print("❌ Error: --replay requires --learn", file=sys.stderr)
            return 1

        tape_path = Path(args.replay)
        if not tape_path.exists():
            print(f"❌ Error: Tape file not found: {tape_path}", file=sys.stderr)
            return 1

        private_knowledge_path = tape_path.with_suffix(".bin")
        # Ensure the directory exists
        private_knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🧠 Creating agent with knowledge store: {private_knowledge_path}")
        replay_agent = build_agent(private_knowledge_path)

        try:
            stats = replay_tape(tape_path, replay_agent, log_interval=args.log_interval)
            return 0
        except KeyboardInterrupt:
            print("\n⚠️  Replay interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error during replay: {e}", file=sys.stderr)
            return 1
        finally:
            # Clean up memory-mapped arrays before closing
            import gc

            gc.collect()
            replay_agent.close()
            if args.compact:
                try:
                    # Use knowledge_* pattern for the output file
                    output_name = tape_path.stem
                    private_knowledge_path = tape_path.parent / f"knowledge_{output_name}.bin"
                    report = prune_and_compact_store(str(private_knowledge_path))
                    print(
                        f"🗜️  Compaction: kept {report['entries_processed']-report['entries_modified']} / {report['entries_processed']}"
                    )
                except Exception as e:
                    print(f"⚠️  Compaction failed: {e}")

    # Validate compilation mode arguments
    if not args.output:
        # Use default knowledge directory
        knowledge_dir = PROJECT_ROOT / "toys/training/knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(knowledge_dir / "wikipedia_simple.gyro" if args.simple else "wikipedia_full.gyro")
        print(f"📁 Using default output: {args.output}")

    # Determine dataset directory
    dataset_dir = (
        PROJECT_ROOT / "toys/training/wikipedia_simple_data"
        if args.simple
        else PROJECT_ROOT / "toys/training/wikipedia_full_data"
    )

    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    # Find dataset files
    files = sorted(dataset_dir.rglob("*"))
    files = [f for f in files if f.is_file() and (f.suffix == ".txt" or f.suffix == ".gz")]

    if not files:
        print(f"❌ Error: No .txt or .gz files found in {dataset_dir}", file=sys.stderr)
        return 1

    print(f"📚 Found {len(files)} files in {dataset_dir}")

    # Create agent if learning is enabled
    agent: Optional[GyroSI] = None
    if args.learn:
        # Use knowledge_* pattern for the output file
        output_name = Path(args.output).stem
        private_knowledge_path = Path(args.output).parent / f"knowledge_{output_name}.bin"
        # Ensure the directory exists
        private_knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_msg = f"🧠 Creating private agent with knowledge store: " f"{private_knowledge_path}"
        print(knowledge_msg)
        agent = build_agent(private_knowledge_path)

    # Create article iterator with configurable blank line threshold
    articles = iter_wiki_articles(files, blank_line_threshold=args.blank_lines)

    # Compile the stream
    try:
        compile_stream(
            articles=articles,
            output_tape_path=Path(args.output),
            agent=agent,
            limit=args.limit,
            log_interval=args.log_interval,
        )
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during compilation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Clean up agent if created
        if agent and args.compact:
            try:
                # Use knowledge_* pattern for the output file
                output_name = Path(args.output).stem
                priv = Path(args.output).parent / f"knowledge_{output_name}.bin"
                report = prune_and_compact_store(str(priv))
                print(
                    f"🗜️  Compaction: kept {report['entries_processed']-report['entries_modified']} / {report['entries_processed']}"
                )
            except Exception as e:
                print(f"⚠️  Compaction failed: {e}")
        if agent:
            try:
                # Clean up memory-mapped arrays before closing
                import gc

                gc.collect()
                agent.close()
                print("🧹 Agent closed successfully")
            except Exception as e:
                print(f"⚠️  Error closing agent: {e}")


if __name__ == "__main__":
    sys.exit(main())
