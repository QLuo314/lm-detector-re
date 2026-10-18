"""Python interface to Ikaleio/lm-detector's official lmfpd CLI."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse

__all__ = ["detect_model", "DetectionError"]


class DetectionError(RuntimeError):
    """Detection failed; report may contain successful rounds and failure details."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


def detect_model(
    base_url: str,
    api_key: str,
    model: str,
    *,
    api: str = "responses",
    repeat: int = 1,
    count: int = 3,
    parallel: int = 3,
    timeout: float = 120,
    total_timeout: float = 600,
    strict: bool = False,
    stream: bool = True,
    tokenizer: bool = False,
    effort: str = None,
) -> dict:
    """Probe an API and return lmfpd's complete JSON report unchanged.

    Requires Node.js >=22 and `npm ci` in this package's directory.
    Sends count * repeat fingerprint requests; tokenizer=True adds probes.
    timeout is the upstream per-request timeout (first byte for streaming).
    total_timeout bounds the entire CLI process, including streaming.
    A nonzero exit raises DetectionError, with any JSON report on .report.
    Rankings compare reference-bank candidates; they are not proof of identity.
    """
    for name, value in (("base_url", base_url), ("api_key", api_key), ("model", model)):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{name} must be a nonempty string without NUL characters")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base_url must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Pass credentials via api_key, not URL userinfo/query/fragment")
    if api not in ("responses", "chatcompletion", "cc", "message"):
        raise ValueError("api must be responses, chatcompletion, cc, or message")
    for name, value in (("repeat", repeat), ("count", count), ("parallel", parallel)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if count > 3 or parallel > 3:
        raise ValueError("count and parallel must be in 1..3")
    if strict and count != 3:
        raise ValueError("strict mode requires count=3")
    for name, value in (("timeout", timeout), ("total_timeout", total_timeout)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    for name, value in (("strict", strict), ("stream", stream), ("tokenizer", tokenizer)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if effort is not None and (not isinstance(effort, str) or not effort.strip() or "\x00" in effort):
        raise ValueError("effort must be a nonempty string without NUL characters")

    node = shutil.which("node")
    package = Path(__file__).resolve().parent / "node_modules" / "lmfpd"
    if not node:
        raise DetectionError("Node.js >=22 is required")
    try:
        metadata = json.loads((package / "package.json").read_text())
        entry = metadata["bin"]
        entry = entry["fpd"] if isinstance(entry, dict) else entry
    except (OSError, ValueError, KeyError, TypeError):
        raise DetectionError(f"Install the CLI first: npm ci --prefix {package.parent.parent}") from None

    command = [node, str(package / entry), "--json", "--no-update-check",
               "--api", api, "--repeat", str(repeat), "--count", str(count),
               "--parallel", str(min(parallel, count)), "--timeout", str(timeout)]
    for enabled, flag in ((strict, "--strict"), (not stream, "--no-stream"), (tokenizer, "--tokenizer")):
        if enabled:
            command.append(flag)
    if effort is not None:
        command.extend(["--effort", effort])
    # Credentials never enter command arguments; the parent environment is unchanged.
    env = dict(os.environ, API_KEY=api_key, BASE_URL=base_url, MODEL=model,
               NO_COLOR="1", FPD_NO_UPDATE_CHECK="1")
    try:
        completed = subprocess.run(command, env=env, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace",
                                   timeout=total_timeout, check=False)
    except subprocess.TimeoutExpired:
        raise DetectionError(f"Detection exceeded total_timeout={total_timeout}s") from None
    except OSError:
        raise DetectionError("Could not start the lmfpd process") from None
    # Redact echoed credentials before exposing upstream diagnostics or report data.
    stdout = completed.stdout.replace(api_key, "[REDACTED]")
    try:
        report = json.loads(stdout)
    except ValueError:
        report = None
    if completed.returncode:
        detail = completed.stderr.replace(api_key, "[REDACTED]").strip()[-2000:]
        raise DetectionError(f"lmfpd exited with code {completed.returncode}: {detail}", report)
    if not isinstance(report, dict):
        raise DetectionError("lmfpd did not return a JSON object", report)
    return report
