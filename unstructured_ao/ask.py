# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default route from a question to a grounded answer.

    query → chunks → the attributes those chunks are readings of → answer

Chunk search first, with the ontology hung off whatever it found rather than
placed in front of it. That order is the measured one. Starting from concepts
instead caps what can be delivered: only chunks linked to the attributes a
question happens to match are reachable, 72.5% of what chunk search delivers,
and ordering those candidates by question similarity already extracts all of
it — the rest is unreachable, not mis-ranked, so no retrieval change on that
side recovers it. An agent looping over concept search is the same ceiling with
more calls.

What the ontology adds here is annotation. Each passage is labelled with the
indicators it is a reading of, so a figure arrives attributed rather than
bare. On the 318-question benchmark that neither helped nor hurt — 0.5998
against the unannotated control's 0.5991, inside the generator's own sampling
variance — which is the honest state of it: the labels cost nothing and have
not yet been shown to pay.

The relation note is gathered on every call and withheld unless asked for.
It is a different claim from the labels — a label says what a passage is a
reading of, the note says how the readings relate, including which of them are
rival readings of one quantity — and measuring the two together would leave
neither attributable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from unstructured_ao import answer as answer_module
from unstructured_ao import search


@dataclass
class Answer:
    """An answer and everything it was built from.

    *error* is set instead of *text* when the generator could not be reached;
    it is returned rather than raised so a caller answering many questions does
    not lose the batch to one of them.
    """

    text: str = ""
    error: str | None = None
    passages: list[search.Passage] = field(default_factory=list)
    #: One line per passage, naming the indicators it is a reading of. Aligned
    #: with *passages* and empty where the ontology knows nothing about one.
    labels: list[str] = field(default_factory=list)
    #: How those indicators relate. Gathered whether or not it was shown.
    note: str = ""
    #: Whether *note* was put in front of the context.
    note_shown: bool = False

    @property
    def chunks(self) -> list[str]:
        return [p.text for p in self.passages]

    @property
    def pages(self) -> list[tuple[str | None, int | None]]:
        """``(document, page)`` per passage, for citing or scoring."""
        return [(p.document, p.page) for p in self.passages]


def ask(query: str, *, k: int = 10, note: bool = False) -> Answer:
    """Answer *query* from the corpus, annotated with the ontology.

    *k* passages are retrieved and all of them are answered from. *note* also
    puts the relation note in front of the context; it is gathered and returned
    either way.
    """
    passages = search.chunk_search(query, k=k)
    if not passages:
        return Answer(error="no_passages_retrieved")

    found = search.chunk_ontology([p.text for p in passages])
    labels = ["; ".join(names) for names in found.labels]
    # The label belongs to the passage it describes, so it travels on it rather
    # than in a parallel list the caller has to keep aligned.
    passages = [
        search.Passage(
            text=p.text,
            attribute_id=p.attribute_id,
            label=label,
            source=p.source,
            score=p.score,
            document=p.document,
            page=p.page,
        )
        for p, label in zip(passages, labels)
    ]

    text, error = answer_module.answer(
        query,
        [p.text for p in passages],
        labels=labels,
        note=found.note if note else None,
    )
    return Answer(
        text=text,
        error=error,
        passages=passages,
        labels=labels,
        note=found.note,
        note_shown=bool(note and found.note),
    )
