from dataclasses import dataclass

from auto_ontology.retrieval.kumo.predictor import (
    _deduplicate_inferred_links,
)


@dataclass(frozen=True)
class _Column:
    name: str


@dataclass(frozen=True)
class _Table:
    primary_key: _Column


@dataclass(frozen=True)
class _Edge:
    src_table: str
    fkey: str
    dst_table: str


class _Graph:
    def __init__(self) -> None:
        self.edges = [
            _Edge("JOB_OUTCOMES", "restart_of_job_id", "JOBS"),
            _Edge("JOB_OUTCOMES", "job_id", "JOBS"),
            _Edge("JOBS", "project_id", "PROJECTS"),
        ]
        self.tables = {
            "JOBS": _Table(_Column("job_id")),
            "PROJECTS": _Table(_Column("project_id")),
        }

    def __getitem__(self, table: str) -> _Table:
        return self.tables[table]

    def unlink(self, src_table: str, fkey: str, dst_table: str) -> None:
        self.edges.remove(_Edge(src_table, fkey, dst_table))


def test_deduplicate_inferred_links_prefers_destination_primary_key_name() -> None:
    graph = _Graph()

    assert _deduplicate_inferred_links(graph) == 1
    assert graph.edges == [
        _Edge("JOB_OUTCOMES", "job_id", "JOBS"),
        _Edge("JOBS", "project_id", "PROJECTS"),
    ]
