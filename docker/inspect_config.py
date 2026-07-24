#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation" / "harness"))

from eval_config import active_model, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an authoritative runtime configuration without modifying it."
        )
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    model = active_model(config)
    server = config["server"]
    client_host = server["host"]
    if client_host in {"0.0.0.0", "::", "[::]"}:
        client_host = "127.0.0.1"
    if ":" in client_host and not client_host.startswith("["):
        client_host = f"[{client_host}]"
    server_url = "http://{}:{}".format(client_host, server["port"])
    review_dedup = config.get("review_dedup")
    review_enabled = bool(review_dedup and review_dedup["enabled"])
    review_backend = (
        str(review_dedup.get("backend", "voyage"))
        if review_enabled
        else None
    )
    review_auto_start = bool(
        review_enabled
        and review_backend == "voyage"
        and review_dedup.get("auto_start", False)
    )
    review_url = review_dedup["base_url"] if review_auto_start else None
    review_parsed = urlparse(review_url) if review_url else None
    review_health_url = (
        f"{review_parsed.scheme}://{review_parsed.netloc}/health"
        if review_parsed
        else None
    )
    print(
        json.dumps(
            {
                "server_host": server["host"],
                "server_port": server["port"],
                "server_url": server_url,
                "expected_gpu_count": (
                    model.tensor_parallel_size * model.data_parallel_size
                ),
                "target_model": str(model.target),
                "draft_model": str(model.draft) if model.draft else None,
                "review_dedup_enabled": review_enabled,
                "review_dedup_backend": review_backend,
                "review_dedup_auto_start": review_auto_start,
                "review_dedup_model": (
                    review_dedup["model"] if review_auto_start else None
                ),
                "review_dedup_port": (
                    review_parsed.port if review_parsed else None
                ),
                "review_dedup_base_url": review_url,
                "review_dedup_health_url": review_health_url,
            }
        )
    )


if __name__ == "__main__":
    main()
