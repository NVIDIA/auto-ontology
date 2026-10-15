// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Role } from '@/enums/auth';

/** Longest group name the API accepts, counted after trimming. */
export const MAX_GROUP_NAME_LENGTH = 100;

/** A role carried by a group. The same two an account has. */
export type GroupRoleName = Role;

export type GroupMember = {
	id: string;
	name: string;
	email: string;
	role: string | null;
};

/** A group as `GET /api/groups` returns it, including the permissions of its role. */
export type Group = {
	id: string;
	name: string;
	description: string;
	color: string;
	role: GroupRoleName;
	roles: { id: string; name: string }[];
	users: GroupMember[];
	permissions: string[];
	created: string;
};

export type GroupWriteInput = {
	name: string;
	description: string | null;
	color: string;
	role: GroupRoleName;
	user_ids: string[];
};
