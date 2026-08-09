// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelFormat } from '@/enums/modelInterchange';

/** Per-entity counts returned by POST /api/model/import (see gsf/dal/model_interchange.py). */
export type ImportEntityCounts = {
	databases: number;
	schemas: number;
	tables: number;
	columns: number;
	terms: number;
	column_attributes: number;
	sql_attributes: number;
	custom_analyses: number;
};

export type ImportEmbedCounts = {
	data_rows: number;
	semantic_rows: number;
};

export type ImportEmbedResult = ImportEmbedCounts & {
	skipped: boolean;
	reason?: string;
};

export type ImportSummary = {
	/** Format the backend read the uploaded file as. */
	format: ModelFormat;
	database_ids: string[];
	created: ImportEntityCounts;
	skipped: ImportEntityCounts;
	replace: boolean;
	terms: number;
	column_attributes: number;
	semantic_fks: number;
	sql_attributes: number;
	custom_analyses: number;
	pending_embed?: ImportEmbedCounts;
	embeddings?: ImportEmbedResult;
};

export type ImportModelResult = {
	success: boolean;
	summary: ImportSummary;
};
