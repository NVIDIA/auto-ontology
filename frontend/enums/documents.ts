// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Where an uploaded document is in ingestion. Mirrors `DocumentStatus` in
 * `auto_ontology/server/models.py`.
 */
export enum DocumentStatus {
	Processing = 'processing',
	/** Chunks are stored; the term and attributes are being extracted. */
	Extracting = 'extracting',
	Ready = 'ready',
	Failed = 'failed',
}
