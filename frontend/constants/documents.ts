// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Mirrors `MAX_UPLOAD_BYTES` in `auto_ontology/server/documents/router.py`, which rejects larger. */
export const MAX_DOCUMENT_UPLOAD_BYTES = 200 * 1024 * 1024;

/**
 * Mirrors `MAX_DOCUMENT_WORDS` in `unstructured_ao/config.py`. Only shown: words
 * are counted after extraction, so a file over it is accepted and turns `failed`.
 */
export const MAX_DOCUMENT_WORDS = 500_000;

/** Files one upload dialog sends. Each goes as its own request. */
export const MAX_DOCUMENT_FILES_PER_UPLOAD = 10;

/**
 * Extensions the backend's default (`auto`) extractor reads. Mirrors
 * `_AUTO_EXTRACTORS` in `unstructured_ao/ingest.py`; the upload route is the
 * authority and answers 400 for anything else.
 */
export const SUPPORTED_DOCUMENT_EXTENSIONS = ['pdf', 'docx', 'pptx', 'html', 'txt'] as const;

/** How often the list re-reads while a document is still being ingested. */
export const DOCUMENT_STATUS_POLL_MS = 5000;
