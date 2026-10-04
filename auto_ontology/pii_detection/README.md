<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PII Detection

Rules-first PII classification for database columns. During catalog ingestion,
decisions are persisted through the project's existing `PII` tag and the
`pii_processed` marker; detection remains independently testable from storage.

## Flow

1. High-confidence rules handle obvious PII and obvious non-PII.
2. Uncertain columns are sent to an optional LLM backend and conservatively
   treated as PII.
3. Results include status, category, confidence, reason, and source.
4. Positive results above the chosen confidence threshold are auto-tagged.
   Positive results below it are left unprocessed, so the column stays
   blocked and is reclassified on the next ingestion.
5. Profiling and live probes fail closed: they can read values only from
   classified columns that do not carry `PII`.
6. Applying `PII` deletes persisted samples and any data or semantic embedding
   rows that may contain them. Probe audit entries retain purpose, counts, and
   timing, but never SQL, errors, or result values.
7. Persisted non-PII samples carry an update timestamp. Expired values and
   sample-bearing embeddings are deleted before semantic compilation (30 days
   by default, configurable up to one year).

## Usage

```python
from auto_ontology.pii_detection import ColumnInput, PiiDetector

detector = PiiDetector()
decision = detector.detect(
    ColumnInput(column_name="email_address", table_name="customers")
)

if decision.should_auto_tag(threshold=0.9):
    # The ingestion integration attaches the shared PII tag to this column.
    ...
```

For uncertain columns, the ingestion integration uses the project's configured
LLM through `LlmPiiClassifier`:

```python
from auto_ontology.pii_detection import LlmPiiClassifier, PiiDetector

detector = PiiDetector(LlmPiiClassifier())
```

Raw sample values from PII or not-yet-classified columns are never selected,
persisted, embedded, or included in model prompts.
