# Contributing to Auto Ontology

Thank you for your interest in contributing to Auto Ontology!
Contributions of all kinds are welcome: bug reports, feature requests,
documentation improvements, and code.

## Reporting Issues

- Search [existing issues](https://github.com/NVIDIA/auto-ontology/issues) before filing a
  new one.
- **Do not report security vulnerabilities through GitHub issues.** See
  [SECURITY.md](SECURITY.md) for NVIDIA's coordinated disclosure process.

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

1. Fork the repository and create your branch from `main`.
2. Add tests for new functionality and make sure the full test suite passes.
3. Keep pull requests focused — one logical change per PR.
4. Pull requests targeting `main` require **an approving review from a code
   owner** before they can be merged. Code owners (see `.github/CODEOWNERS`)
   are automatically requested for review. Because a code owner cannot
   approve their own pull request, one of the other owners must review it.

## Developer Certificate of Origin (DCO)

All contributions must be signed off to certify that you wrote the change or
otherwise have the right to submit it under the project license, per the
[Developer Certificate of Origin](https://developercertificate.org/).

Add a `Signed-off-by` line to every commit (use `git commit -s`):

```
Signed-off-by: Your Name <your.email@example.com>
```

Commits without a valid sign-off cannot be accepted.

## License

By contributing to Auto Ontology, you agree that your contributions will be licensed
under the [Apache License 2.0](LICENSE).
