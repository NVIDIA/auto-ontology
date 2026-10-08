# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unstructured document ingest pipeline for auto-ontology.

End-to-end pipeline for a single document
------------------------------------------
    import unstructured.config  # Ray override — import before nemo_retriever
    from unstructured.ingest import ingest_file
    from unstructured.extract import run_extraction
    from unstructured.retrieve import link_document_attributes
    from auto_ontology.dal.session import get_engine

    engine = get_engine()
    with engine.begin() as conn:
        doc_id = ingest_file("path/to/file.pdf", conn)
        if doc_id:
            run_extraction(doc_id, full_text, conn)
            link_document_attributes(doc_id, conn)

See each module's docstring for details.
"""
