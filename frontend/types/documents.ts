// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DocumentStatus } from '@/enums/documents';

export type DocumentItem = {
	id: string;
	name: string;
	description: string | null;
	/** Extension without the dot, e.g. `pdf`. */
	file_type: string;
	status: DocumentStatus;
	/** Why ingestion failed; set only when `status` is `failed`. */
	error: string | null;
	/** ISO 8601 — when the file was added. */
	ingested_at: string;
};

/** Fields left out are left as they are; an empty description clears it. */
export type DocumentUpdateInput = {
	name?: string;
	description?: string;
};
