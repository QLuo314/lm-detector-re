"""Run from the workspace root: python3 -m lm_detector_function.example.

Setup after copying this folder: npm ci --prefix lm_detector_function
Requires Python >=3.9 and Node.js >=22; no third-party Python packages.
Set BASE_URL, API_KEY and MODEL in your environment before running.
Upstream: https://github.com/Ikaleio/lm-detector (MIT).
CLI version is pinned in package-lock.json for reproducibility.
"""
import json
import os

from lm_detector_function import detect_model


if __name__ == "__main__":
    result = detect_model(
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["API_KEY"],
        model=os.environ["MODEL"],
        api="cc",  # responses / cc / message
        repeat=1,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
