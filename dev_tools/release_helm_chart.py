# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package ``helm/auto-ontology`` and publish it to the NGC Helm chart registry.

Publishes ``<org>[/<team>]/<name>:<version>``, served from
``https://helm.ngc.nvidia.com/<org>[/<team>]/charts/``. Before packaging,
checks that the chart's ``values.yaml`` pulls the images release-docker.yml
publishes -- ``nvcr.io/<org>[/<team>]/<image>:<app-version>`` -- so a release
can't ship a chart that installs some other build.

Usage (from the repo root)::

    helm lint helm/auto-ontology
    python dev_tools/release_helm_chart.py \\
        --org 0966117469611503 --version 1.0.0 --app-version 1.0 [--dry-run]

Requires ``pip install ngcsdk pyyaml``; ``NGC_CLI_API_KEY`` is needed to publish.
Run by ``.github/workflows/release-helm.yml``.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"

# values.yaml key -> the image release-docker.yml publishes for it.
CHART_IMAGES = {
    "backend": "auto-ontology",
    "ingestion": "auto-ontology",
    "frontend": "auto-ontology-frontend",
}

_NOT_FOUND_EXC = frozenset({"ResourceNotFoundException", "ChartNotFoundException"})
_ALREADY_EXISTS_EXC = frozenset(
    {"ResourceAlreadyExistsException", "ChartAlreadyExistsException"}
)


def _exc_name(exc: BaseException) -> str:
    return type(exc).__name__


def _namespace(args: argparse.Namespace) -> str:
    return f"{args.org}/{args.team}" if args.team else args.org


def _check_images(values: dict, registry: str, app_version: str) -> list[str]:
    errors = []
    for key, image in CHART_IMAGES.items():
        got = values.get(key, {}).get("image", {})
        want_repo = f"{registry}/{image}"
        if got.get("repository") != want_repo:
            errors.append(
                f"{key}.image.repository is {got.get('repository')!r}, "
                f"expected {want_repo!r}"
            )
        # Compared uncoerced: an unquoted `tag: 1.0` is the float 1.0, which
        # Helm renders as `:1` -- a tag that was never pushed.
        if got.get("tag") != app_version:
            errors.append(
                f"{key}.image.tag is {got.get('tag')!r}, expected {app_version!r}"
                " (quoted)"
            )
    return errors


def _stage(chart_dir: pathlib.Path, name: str, version: str, app_version: str) -> None:
    staged = DIST_DIR / name
    shutil.rmtree(staged, ignore_errors=True)
    shutil.copytree(chart_dir, staged)
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    )
    (staged / ".gitsha").write_text(sha)

    chart_yaml = staged / "Chart.yaml"
    chart = yaml.safe_load(chart_yaml.read_text())
    chart["name"] = name
    chart["version"] = version
    chart["appVersion"] = app_version
    chart_yaml.write_text(yaml.safe_dump(chart, sort_keys=False))


def _publish(args: argparse.Namespace, tgz: pathlib.Path) -> None:
    api_key = os.environ.get("NGC_CLI_API_KEY", "")
    if not api_key:
        sys.exit("ERROR: NGC_CLI_API_KEY environment variable is not set")

    from ngcsdk import Client

    clt = Client()
    clt.configure(api_key=api_key, org_name=args.org, team_name=args.team or None)

    target = f"{_namespace(args)}/{args.name}"
    metadata_kwargs = dict(
        short_description=args.description,
        display_name=args.display_name,
        publisher="NVIDIA",
    )
    # The NGC page's overview tab; the chart has no README of its own yet.
    overview = pathlib.Path(args.chart_dir) / "README.md"
    if overview.is_file():
        metadata_kwargs["overview_filepath"] = str(overview)

    print(f"Updating chart metadata for {target} ...")
    try:
        clt.registry.chart.update(target=target, **metadata_kwargs)
    except Exception as exc:
        if _exc_name(exc) not in _NOT_FOUND_EXC:
            raise
        print(f"Chart '{target}' not found ({_exc_name(exc)}); creating it ...")
        clt.registry.chart.create(target=target, **metadata_kwargs)

    print(f"Pushing chart {target}:{args.version} ...")
    try:
        clt.registry.chart.push(
            target=f"{target}:{args.version}", source_dir=str(tgz.parent)
        )
    except Exception as exc:
        if _exc_name(exc) not in _ALREADY_EXISTS_EXC:
            raise
        # Failed rather than skipped (as Retriever does): a run that keeps the
        # old contents and reports success reads as a release that happened.
        sys.exit(
            f"ERROR: chart version {args.version!r} already exists in NGC "
            f"({_exc_name(exc)}); release it under a new version"
        )
    print(f"Successfully pushed {target}:{args.version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--org", required=True, help="Target NGC org")
    parser.add_argument(
        "-t", "--team", default="", help="Target NGC team; omit for org level"
    )
    parser.add_argument("-v", "--version", required=True, help="Chart version")
    parser.add_argument(
        "-a", "--app-version", required=True, help="Image tag the chart deploys"
    )
    parser.add_argument("-n", "--name", default="auto-ontology", help="Chart name")
    parser.add_argument(
        "--chart-dir", default=str(REPO_ROOT / "helm" / "auto-ontology")
    )
    parser.add_argument("--display-name", default="NVIDIA Auto Ontology Helm Chart")
    parser.add_argument(
        "-d",
        "--description",
        default="Auto Ontology: FastAPI backend, ingestion worker, Next.js "
        "frontend and Postgres (pgvector)",
    )
    parser.add_argument("-r", "--dry-run", action="store_true")
    args = parser.parse_args()

    chart_dir = pathlib.Path(args.chart_dir)
    if not chart_dir.is_dir():
        sys.exit(f"ERROR: chart directory does not exist: {chart_dir}")

    values = yaml.safe_load((chart_dir / "values.yaml").read_text())
    registry = f"nvcr.io/{_namespace(args)}"
    errors = _check_images(values, registry, args.app_version)
    if errors:
        print("ERROR: values.yaml does not match this release:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)

    _stage(chart_dir, args.name, args.version, args.app_version)
    subprocess.check_call(
        ["helm", "package", str(DIST_DIR / args.name), "--destination", str(DIST_DIR)]
    )
    tgz = DIST_DIR / f"{args.name}-{args.version}.tgz"
    if not tgz.is_file():
        sys.exit(f"ERROR: helm package did not produce {tgz}")

    if args.dry_run:
        print(f"[DRY RUN] Chart packaged: {tgz}")
        print(
            f"[DRY RUN] Skipping NGC push of {_namespace(args)}/{args.name}:{args.version}"
        )
        return
    _publish(args, tgz)


if __name__ == "__main__":
    main()
