from __future__ import annotations

from auto_ontology.vdb.postgres import PostgresVDB


def test_vector_size_reads_embedding_dimensions_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")

    store = PostgresVDB(
        connection_string="postgresql://user:password@localhost:5432/db",
    )

    assert store.vector_size == 1024
