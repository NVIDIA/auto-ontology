# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PostgresVDB`` must satisfy the upstream ``VDB`` ABC.

This is a dependency-contract test, not a behaviour test, and it exists because
the suite could not catch the failure it guards. ``nemo-retriever`` added ten
abstract methods to ``VDB``; ``PostgresVDB`` did not grow them, so it became
abstract and *every* ``get_data_vdb()`` raised ``TypeError`` at construction.
Nothing failed in CI, because the tests that touch a VDB mock it — the break
only appeared when a real one was built, on the ingest and search paths.

Verifying that the imported *symbols* still exist is not enough for a
dependency bump. Verify the contract.
"""

from __future__ import annotations

import inspect

from nemo_retriever.common.vdb.adt_vdb import VDB

from gsf.vdb.postgres import PostgresVDB


def test_postgres_vdb_is_concrete() -> None:
    missing = sorted(getattr(PostgresVDB, "__abstractmethods__", ()))
    assert not missing, (
        f"PostgresVDB is abstract and cannot be constructed; unimplemented: "
        f"{missing}. Every get_data_vdb()/get_semantic_vdb() call raises "
        f"TypeError until these exist."
    )


def test_the_methods_gsf_actually_uses_are_real_implementations() -> None:
    """The stubs are deliberate; these must never become one.

    ``run`` and ``write_to_index`` carry ingestion, ``retrieval`` carries
    search, and the three deletes carry reset. A stub here would be a silent
    data path, not a loud unsupported one.
    """
    for name in (
        "run",
        "write_to_index",
        "retrieval",
        "create_index",
        "delete_by_id",
        "delete_by_database",
        "delete_all",
    ):
        method = getattr(PostgresVDB, name, None)
        assert method is not None, f"{name} is missing"
        source = inspect.getsource(method)
        assert "_unsupported" not in source, (
            f"{name}() is a stub, but GSF calls it on a live path"
        )


def test_every_abstract_method_is_accounted_for() -> None:
    """Either implemented here, or inherited as a concrete upstream default."""
    for name in sorted(VDB.__abstractmethods__):
        if name == "__init__":
            continue
        assert hasattr(PostgresVDB, name), f"{name} is not defined anywhere"
