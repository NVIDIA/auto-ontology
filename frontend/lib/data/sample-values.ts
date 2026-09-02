// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Client-side mirror of `gsf.utils.sample_values.parse_sample_values` — most
 * of the API already runs a Column's `sample_values` through that helper
 * before responding, but `ForeignKeyRef.source_sample_values`/
 * `target_sample_values` (see `gsf/server/models.py`) come straight off the
 * Column row exactly as stored: profiling persists them as a
 * JSON string, a catalog PATCH stores a real array, so either shape can
 * reach the client for that one field. Normalizes both to a plain string
 * array (or `null`) so callers never have to branch on which one they got.
 */
export const parseSampleValues = (raw: string[] | string | null | undefined): string[] | null => {
	if (raw == null) return null;
	if (Array.isArray(raw)) return raw.map(String);
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch {
		return null;
	}
	if (!Array.isArray(parsed)) return null;
	return parsed.map(String);
};
