// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAccessControl } from 'better-auth/plugins/access';
import { adminAc, defaultStatements } from 'better-auth/plugins/admin/access';
import { Role } from '@/enums/auth';
import access from './access.json';

/**
 * Access-control statements + roles shared by the Better Auth server and client.
 * `admin` and `viewer` are the only roles. The Auto Ontology resources and the
 * actions each role gets live in `access.json`, which the Python API reads too
 * when it expands a group's role. On top of that vocabulary we keep Better
 * Auth's built-in `user` / `session` resources (`defaultStatements`, and
 * `adminAc.statements` for the admin role). Those govern account administration
 * and are not in the shared file: a group does not grant them.
 *
 * Every protected API/page authorizes against these roles (see
 * `auth/permissions.ts`).
 *
 * Viewer is a subset. Reading and changing groups stay with admins, because a
 * group lists its members' names and emails. Viewers may read zones, tags, and
 * connections, and may edit the catalog. Glossary, prompts, and custom analyses
 * are fully available to viewers so those pages do not 403.
 */
const statement = {
	...defaultStatements,
	...access.admin,
} as const;

export const ac = createAccessControl(statement);

export const roles = {
	[Role.Admin]: ac.newRole({
		...adminAc.statements,
		...access.admin,
	}),
	[Role.Viewer]: ac.newRole({
		...access.viewer,
	}),
};

/**
 * Shape accepted by a role's `authorize()` — a partial map of resource → the
 * actions being requested, e.g. `{ analytics: ['read'] }`. Both roles share the
 * same access-control statements, so either role's request type works.
 */
export type PermissionRequest = Parameters<(typeof roles)[Role.Admin]['authorize']>[0];
