// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type Zone = {
	id: string;
	name: string;
	label: string;
	description: string | null;
	color: string | null;
	enabled: boolean;
};

export type ZoneItem = {
	id: string;
	name: string;
	label: string;
};

export type ZoneDetail = Zone & {
	items: ZoneItem[];
};

/** Returned by POST /zones — items is a list of linked IDs, not full objects. */
export type ZoneCreated = Zone & {
	items: string[];
};

export type ZoneCreateInput = {
	name: string;
	description?: string;
	color: string;
	items: string[];
	created_by: string;
};

export type ZoneUpdateInput = {
	name?: string;
	description?: string | null;
	color?: string | null;
	items?: string[];
};
