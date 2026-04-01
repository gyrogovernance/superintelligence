from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_root_s = str(_REPO_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)

from src.tools.gyroscopic.config import get_gyroscopic_llm_config, repo_root
from src.tools.gyroscopic.loader import build_llama_cli_command

THROUGHPUT_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]"
)

TRACE_RE = re.compile(
    r"GyroMatMul trace:\s*vec_dot_f32=(\d+)\s*vec_dot_q8_0_q8_0=(\d+)\s*gemm_q8_0=(\d+)"
)

PROMPTS = [
    "Hello",
    "Write a short paragraph about integer arithmetic.",
    "List five properties of deterministic computation.",
    "Explain why exact replay matters in systems engineering.",
]

INCLUDE_REF = os.environ.get("GYRO_BENCH_INCLUDE_REF", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

BENCH_MODES = (
    ("stock", "gyro_ref", "gyro_avx2") if INCLUDE_REF else ("stock", "gyro_avx2")
)

SMOKE = os.environ.get("GYRO_BENCH_SMOKE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

MAX_PROMPTS_ENV = os.environ.get("GYRO_BENCH_MAX_PROMPTS", "").strip()

_DEFAULT_N_PREDICT = 128
_DEFAULT_TIMEOUT_SEC = 1800.0
_DEFAULT_POLL_SEC = 2.0

DEFAULT_TIMEOUT_SEC = float(os.environ.get("GYRO_BENCH_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT_SEC)))
POLL_SEC = float(os.environ.get("GYRO_BENCH_POLL_SEC", str(_DEFAULT_POLL_SEC)))
VERBOSE_PROGRESS = os.environ.get("GYRO_BENCH_VERBOSE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def n_predict() -> int:
    return int(os.environ.get("GYRO_BENCH_N_PREDICT", str(_DEFAULT_N_PREDICT)))


@dataclass
class BenchRun:
    mode: str
    prompt: str
    returncode: int | None
    timed_out: bool
    elapsed_sec: float
    prompt_tps: float | None
    gen_tps: float | None
    stdout_sha256: str
    stdout_path: str
    stderr_path: str
    hook_trace_seen: bool
    unsupported_seen: bool
    gyroscopic_banner_seen: bool
    vec_dot_f32_calls: int | None
    vec_dot_q80_calls: int | None
    gemm_q8_0_calls: int | None

    @property
    def is_gyro(self) -> bool:
        return self.mode.startswith("gyro_")


def normalize_stdout(text: str) -> str:
    out = []
    for line in text.splitlines():
        if THROUGHPUT_RE.search(line):
            continue
        if line.startswith("llama_memory_breakdown_print:"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def parse_throughput(text: str) -> tuple[float | None, float | None]:
    m = THROUGHPUT_RE.search(text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def avg(xs: list[float | None]) -> float | None:
    ys = [x for x in xs if x is not None]
    if not ys:
        return None
    return sum(ys) / len(ys)


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def run_case(mode: str, prompt: str, run_idx: int, out_dir: Path, n_pred: int) -> BenchRun:
    if mode not in BENCH_MODES:
        raise ValueError(f"mode must be one of {BENCH_MODES}, got {mode!r}")

    cfg = get_gyroscopic_llm_config()
    argv = build_llama_cli_command(
        cfg,
        prompt=prompt,
        n_predict=n_pred,
        extra_args=[
            "--seed", "123",
            "--temp", "0",
            "--top-k", "1",
            "--top-p", "1.0",
            "--repeat-penalty", "1.0",
        ],
    )

    env = dict(os.environ)
    if mode == "stock":
        env["GGML_GYROSCOPIC_MATMUL"] = "0"
        env["GGML_GYROSCOPIC_TRACE"] = "0"
        env["GGML_GYROSCOPIC_STRICT"] = "1"
        env.pop("GGML_GYROSCOPIC_KERNEL", None)
    elif mode == "gyro_ref":
        env["GGML_GYROSCOPIC_MATMUL"] = "1"
        env["GGML_GYROSCOPIC_TRACE"] = "1"
        env["GGML_GYROSCOPIC_STRICT"] = "1"
        env["GGML_GYROSCOPIC_KERNEL"] = "ref"
    else:
        env["GGML_GYROSCOPIC_MATMUL"] = "1"
        env["GGML_GYROSCOPIC_TRACE"] = "1"
        env["GGML_GYROSCOPIC_STRICT"] = "1"
        env["GGML_GYROSCOPIC_KERNEL"] = "avx2"

    tag = mode
    stdout_path = out_dir / f"{run_idx:02d}_{tag}.stdout.txt"
    stderr_path = out_dir / f"{run_idx:02d}_{tag}.stderr.txt"

    actual_timeout = (
        DEFAULT_TIMEOUT_SEC
        if mode != "gyro_ref"
        else min(DEFAULT_TIMEOUT_SEC, 120.0)
    )

    print(
        f"\n=== RUN {run_idx:02d} | {tag} ===\n"
        f"prompt={prompt!r}\n"
        f"n_predict={n_pred}\n"
        f"timeout_sec={actual_timeout}\n"
        f"exe={argv[0]}\n"
    )

    start = time.perf_counter()
    timed_out = False
    returncode: int | None = None

    with stdout_path.open("w", encoding="utf-8", errors="replace") as fout, \
         stderr_path.open("w", encoding="utf-8", errors="replace") as ferr:

        proc = subprocess.Popen(
            argv,
            stdout=fout,
            stderr=ferr,
            stdin=subprocess.DEVNULL,
            env=env,
            text=True,
        )

        while True:
            rc = proc.poll()
            elapsed = time.perf_counter() - start

            if rc is not None:
                returncode = rc
                break

            if elapsed > actual_timeout:
                timed_out = True
                proc.kill()
                returncode = None
                break

            if VERBOSE_PROGRESS:
                print(f"[running] {tag} elapsed={elapsed:.1f}s", flush=True)
            time.sleep(POLL_SEC)

    elapsed = time.perf_counter() - start

    stdout_text = _read_text(stdout_path)
    stderr_text = _read_text(stderr_path)

    stdout_norm = normalize_stdout(stdout_text)
    sha = hashlib.sha256(stdout_norm.encode("utf-8")).hexdigest()
    prompt_tps, gen_tps = parse_throughput(stdout_text)

    hook_trace_seen = "GyroMatMul trace:" in stderr_text
    unsupported_seen = "unsupported live path" in stderr_text
    gyroscopic_banner_seen = "GyroMatMul: compiled=1" in stderr_text

    trace_match = TRACE_RE.search(stderr_text)
    vec_dot_f32_calls = int(trace_match.group(1)) if trace_match else None
    vec_dot_q80_calls = int(trace_match.group(2)) if trace_match else None
    gemm_q8_0_calls = int(trace_match.group(3)) if trace_match else None

    print(
        f"[done] {tag} elapsed={elapsed:.2f}s returncode={returncode} "
        f"timed_out={timed_out} prompt_tps={prompt_tps} gen_tps={gen_tps}"
    )
    print(
        f"[diag] hook_trace_seen={hook_trace_seen} "
        f"unsupported_seen={unsupported_seen} "
        f"gyroscopic_banner_seen={gyroscopic_banner_seen} "
        f"vec_dot_f32_calls={vec_dot_f32_calls} vec_dot_q8_0_calls={vec_dot_q80_calls} "
        f"gemm_q8_0_calls={gemm_q8_0_calls}"
    )

    return BenchRun(
        mode=mode,
        prompt=prompt,
        returncode=returncode,
        timed_out=timed_out,
        elapsed_sec=elapsed,
        prompt_tps=prompt_tps,
        gen_tps=gen_tps,
        stdout_sha256=sha,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        hook_trace_seen=hook_trace_seen,
        unsupported_seen=unsupported_seen,
        gyroscopic_banner_seen=gyroscopic_banner_seen,
        vec_dot_f32_calls=vec_dot_f32_calls,
        vec_dot_q80_calls=vec_dot_q80_calls,
        gemm_q8_0_calls=gemm_q8_0_calls,
    )


def main() -> int:
    root = repo_root()
    out_dir = root / "data" / "benchmarks" / "gyromatmul_llama"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = list(PROMPTS)
    if SMOKE:
        prompts = PROMPTS[:1]

    if MAX_PROMPTS_ENV:
        try:
            k = int(MAX_PROMPTS_ENV)
            if k > 0:
                prompts = prompts[:k]
        except ValueError:
            pass

    npred = n_predict()
    if SMOKE and "GYRO_BENCH_N_PREDICT" not in os.environ:
        npred = 8

    print(
        f"Bench: {len(prompts)} prompt(s), n_predict={npred}, smoke={SMOKE}, "
        f"include_ref={INCLUDE_REF}, {len(BENCH_MODES)} mode(s) per prompt "
        f"(~{len(prompts) * len(BENCH_MODES)} llama-cli runs).\n"
    )

    runs: list[BenchRun] = []
    exact_flags: list[bool] = []
    fatal = False

    for i, prompt in enumerate(prompts):
        if INCLUDE_REF:
            stock_run = run_case("stock", prompt, 3 * i, out_dir, npred)
            ref_run = run_case("gyro_ref", prompt, 3 * i + 1, out_dir, npred)
            avx2_run = run_case("gyro_avx2", prompt, 3 * i + 2, out_dir, npred)
            runs.extend([stock_run, ref_run, avx2_run])

            exact = (
                not stock_run.timed_out
                and not ref_run.timed_out
                and not avx2_run.timed_out
                and stock_run.returncode == 0
                and ref_run.returncode == 0
                and avx2_run.returncode == 0
                and stock_run.stdout_sha256 == ref_run.stdout_sha256 == avx2_run.stdout_sha256
            )
            exact_flags.append(exact)

            print(f"\nPROMPT {i + 1}")
            for label, r in (
                ("stock", stock_run),
                ("gyro_ref", ref_run),
                ("gyro_avx2", avx2_run),
            ):
                print(
                    f"  {label:10} : rc={r.returncode} timeout={r.timed_out} "
                    f"prompt_tps={r.prompt_tps} gen_tps={r.gen_tps} sha={r.stdout_sha256}"
                )
            print(f"  all_three_exact : {exact}")

            gyro_runs = (ref_run, avx2_run)
        else:
            stock_run = run_case("stock", prompt, 2 * i, out_dir, npred)
            avx2_run = run_case("gyro_avx2", prompt, 2 * i + 1, out_dir, npred)
            runs.extend([stock_run, avx2_run])

            exact = (
                not stock_run.timed_out
                and not avx2_run.timed_out
                and stock_run.returncode == 0
                and avx2_run.returncode == 0
                and stock_run.stdout_sha256 == avx2_run.stdout_sha256
            )
            exact_flags.append(exact)

            print(f"\nPROMPT {i + 1}")
            for label, r in (("stock", stock_run), ("gyro_avx2", avx2_run)):
                print(
                    f"  {label:10} : rc={r.returncode} timeout={r.timed_out} "
                    f"prompt_tps={r.prompt_tps} gen_tps={r.gen_tps} sha={r.stdout_sha256}"
                )
            print(f"  stock_vs_avx2_exact : {exact}")

            gyro_runs = (avx2_run,)

        for gr in gyro_runs:
            if gr.timed_out:
                print(f"  ERROR : {gr.mode} timed out")
                fatal = True
            if gr.returncode not in (0, None):
                print(f"  ERROR : {gr.mode} returned nonzero")
                fatal = True
            if gr.unsupported_seen:
                print(f"  ERROR : {gr.mode} hit unsupported live path")
                fatal = True
            if not gr.timed_out and gr.returncode == 0 and not gr.gyroscopic_banner_seen:
                print(
                    f"  ERROR: {gr.mode} missing GyroMatMul: compiled=1 banner; "
                    "not using gyroscopic-enabled ggml or trace off."
                )
                fatal = True
            if not gr.timed_out and gr.returncode == 0 and not gr.hook_trace_seen:
                print(
                    f"  ERROR: {gr.mode} missing GyroMatMul trace line on stderr."
                )
                fatal = True
            if (
                not gr.timed_out
                and gr.returncode == 0
                and (gr.gemm_q8_0_calls is None or gr.gemm_q8_0_calls < 1)
            ):
                print(
                    f"  ERROR: {gr.mode} expected gemm_q8_0 > 0 in trace (mul_mat GEMM path)."
                )
                fatal = True

    def avg_mode(m: str, key: str) -> float | None:
        if key == "prompt":
            return avg([r.prompt_tps for r in runs if r.mode == m and r.returncode == 0])
        return avg([r.gen_tps for r in runs if r.mode == m and r.returncode == 0])

    stock_p = avg_mode("stock", "prompt")
    stock_g = avg_mode("stock", "gen")
    ref_p = avg_mode("gyro_ref", "prompt")
    ref_g = avg_mode("gyro_ref", "gen")
    avx2_p = avg_mode("gyro_avx2", "prompt")
    avx2_g = avg_mode("gyro_avx2", "gen")

    summary = {
        "bench_n_predict": npred,
        "bench_smoke": SMOKE,
        "bench_include_ref": INCLUDE_REF,
        "bench_modes": list(BENCH_MODES),
        "all_exact_match": all(exact_flags),
        "fatal": fatal,
        "stock_prompt_tps_avg": stock_p,
        "stock_gen_tps_avg": stock_g,
        "gyro_ref_prompt_tps_avg": ref_p,
        "gyro_ref_gen_tps_avg": ref_g,
        "gyro_avx2_prompt_tps_avg": avx2_p,
        "gyro_avx2_gen_tps_avg": avx2_g,
        "ref_vs_stock_prompt_ratio": (ref_p / stock_p) if stock_p and ref_p else None,
        "ref_vs_stock_gen_ratio": (ref_g / stock_g) if stock_g and ref_g else None,
        "avx2_vs_stock_prompt_ratio": (avx2_p / stock_p) if stock_p and avx2_p else None,
        "avx2_vs_stock_gen_ratio": (avx2_g / stock_g) if stock_g and avx2_g else None,
        "runs": [asdict(r) for r in runs],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))

    return 2 if fatal else (0 if all(exact_flags) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
