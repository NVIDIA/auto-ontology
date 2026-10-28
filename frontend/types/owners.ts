// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** A user or a group. `owner_id` on an object is one of these ids. */
export type OwnerKind = 'user' | 'group';

export type Owner = {
	id: string;
	name: string;
	kind: OwnerKind;
	/** Set for a group. A hex colour, the same value a group stores. */
	color: string | null;
	/** `resource:action` strings the account's or the group's role grants. */
	permissions: string[];
	/** The signed-in account when it belongs to this group. Empty for a user. */
	member_ids: string[];
};

/** The permission a picker uses to decide who may own that kind of object. */
export const OwnerPermission = {
	catalog: 'catalog:edit',
} as const;
