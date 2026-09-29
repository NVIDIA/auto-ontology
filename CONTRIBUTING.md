<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to Auto Ontology

**This project is currently not accepting external contributions.** Issues,
pull requests, and patches submitted from outside the Auto Ontology maintainer
team will not be reviewed or merged. This matches the
[Contributing](README.md#contributing) section of the README.

**Do not report security vulnerabilities through GitHub issues.** See
[SECURITY.md](SECURITY.md) for NVIDIA's coordinated disclosure process.

The rest of this file is for the Auto Ontology maintainers.

## Development Setup

Auto Ontology uses [uv](https://docs.astral.sh/uv/) for Python dependency management:

```bash
uv sync                # install dependencies
uv run pytest          # run the test suite
uv run ruff check .    # lint
```

See the `Makefile` (`make help`) for build and migration targets, and
`docker-compose.yml` for running the local development stack.

## Pull Requests

1. Create your branch from `main`.
2. Add tests for new functionality and make sure the full test suite passes.
3. Keep pull requests focused — one logical change per PR.
4. Pull requests targeting `main` require **an approving review from a code
   owner** before they can be merged. Code owners (see `.github/CODEOWNERS`)
   are automatically requested for review. Because a code owner cannot
   approve their own pull request, one of the other owners must review it.
5. New files carry the NVIDIA copyright and SPDX license header used across
   the repository.

## License

Auto Ontology is licensed under the [Apache License 2.0](LICENSE).
