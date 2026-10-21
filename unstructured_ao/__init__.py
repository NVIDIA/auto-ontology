# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unstructured documents for auto-ontology: ingest, and answering from them.

Answering a question
--------------------
    from unstructured_ao.ask import ask

    result = ask("what was the temporary contract rate in 2023?")
    print(result.text)

``ask`` is the default route and the measured one: chunk search, then the
attributes those chunks are readings of, then an answer labelled with them.
``unstructured_ao.search`` holds the two retrieval directions on their own —
``chunk_search`` plus ``chunk_ontology`` is what ``ask`` composes, and
``attribute_search`` is the other direction, which is capped below it and kept
because it is what the comparison is against.

End-to-end pipeline for a single document
------------------------------------------
    import unstructured_ao.config  # Ray override — import before nemo_retriever
    from unstructured_ao.dal import get_document_chunks
    from unstructured_ao.ingest import ingest_file
    from unstructured_ao.windows import build_windows
    from unstructured_ao.extract import run_extraction
    from unstructured_ao.retrieve import link_document_attributes
    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.begin() as conn:
        doc_id = ingest_file("path/to/file.pdf", conn)
        if doc_id:
            chunks = get_document_chunks(conn, doc_id)
            windows = build_windows(
                [c["id"] for c in chunks], [c["content"] for c in chunks]
            )
            run_extraction(doc_id, windows, conn)
            link_document_attributes(doc_id, conn)

``ingest_file`` has a CLI (``python -m unstructured_ao.ingest``) and so do
``retrieve`` and ``relate``; the extraction step in the middle does not, and
nothing calls it automatically.  Composing it means building the windows as
above — that is the only thing that constructs the argument ``run_extraction``
takes.

Once the corpus is in, ``unstructured_ao.relate`` adjudicates how the extracted
attributes stand to one another — which of them are rival readings of one
quantity — which is what ``ask`` labels its passages from.

See each module's docstring for details.
"""
