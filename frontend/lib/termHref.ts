// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The address of a term, or of one of its attributes.
 *
 * Terms live on the Exploration page: `semanticId` selects the term's node in
 * the graph and opens its detail beside it, and `sqlAttr`/`colAttr` open one of
 * its attributes in that same panel. Every link to a term goes through here so
 * the graph and the panel always agree on which term is open.
 */
export const termHref = (
	termId: string,
	attribute?: { sqlAttr?: string; colAttr?: string },
): string => {
	const base = `/exploration?semanticId=${encodeURIComponent(termId)}`;
	if (attribute?.sqlAttr != null) {
		return `${base}&sqlAttr=${encodeURIComponent(attribute.sqlAttr)}`;
	}
	if (attribute?.colAttr != null) {
		return `${base}&colAttr=${encodeURIComponent(attribute.colAttr)}`;
	}
	return base;
};
