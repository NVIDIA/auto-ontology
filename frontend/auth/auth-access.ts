// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAccessControl } from 'better-auth/plugins/access';
import { adminAc, defaultStatements } from 'better-auth/plugins/admin/access';
import { Role } from '@/enums/auth';
import access from './access.json';

/**
 * The application's permission vocabulary. Keep the literal tuples here so
 * Better Auth can reject misspelled actions at compile time; JSON imports widen
 * their arrays to `string[]`.
 */
const applicationStatements = {
	analytics: ['read'],
	acronym: ['read', 'create', 'update', 'delete'],
	prompt: ['read', 'create', 'update', 'delete'],
	sso: ['read', 'manage'],
	conversation: ['read', 'write', 'delete'],
	chat: ['use'],
	analysis: ['read', 'manage'],
	catalog: ['read', 'edit'],
	connection: ['read', 'manage'],
	zone: ['read', 'manage'],
	group: ['read', 'manage'],
	tag: ['read', 'manage'],
	semanticCompilation: ['read', 'manage'],
	visualization: ['read', 'manage'],
	pii: ['read', 'manage'],
	modelInterchange: ['export', 'import'],
	apiToken: ['manage'],
} as const satisfies Record<string, readonly string[]>;

type ApplicationStatements = typeof applicationStatements;
type ApplicationRole = {
	[Resource in keyof ApplicationStatements]?: Array<ApplicationStatements[Resource][number]>;
};

/**
 * Check the shared JSON before narrowing its widened `string[]` values back to
 * the literal permission vocabulary.
 */
const validateRoleAccess = (role: Role, roleAccess: Record<string, string[]>): ApplicationRole => {
	for (const [resource, actions] of Object.entries(roleAccess)) {
		const allowedActions = applicationStatements[resource as keyof ApplicationStatements];
		if (!allowedActions) {
			throw new Error(`Unknown permission resource "${resource}" for role "${role}"`);
		}
		for (const action of actions) {
			if (!(allowedActions as readonly string[]).includes(action)) {
				throw new Error(
					`Unknown action "${action}" for resource "${resource}" and role "${role}"`,
				);
			}
		}
	}
	return roleAccess as ApplicationRole;
};

const roleAccess = {
	[Role.Admin]: validateRoleAccess(Role.Admin, access.admin),
	[Role.Viewer]: validateRoleAccess(Role.Viewer, access.viewer),
} satisfies Record<Role, ApplicationRole>;

/**
 * Access-control statements + roles shared by the Better Auth server and client.
 * `admin` and `viewer` are the only roles. Their grants live in `access.json`,
 * which the Python API reads too when it expands a group's role; the matching
 * permission vocabulary is declared above to retain TypeScript's literal
 * action types. On top of that vocabulary we keep Better Auth's built-in
 * `user` / `session` resources (`defaultStatements`, and `adminAc.statements`
 * for the admin role). Those govern account administration and are not in the
 * shared file: a group does not grant them.
 *
 * Every protected API/page authorizes against these roles (see
 * `auth/permissions.ts`).
 *
 * Viewer is a subset. Reading and changing groups stay with admins, because a
 * group lists its members' names and emails. A viewer may read the catalog,
 * zones, tags, connections, glossary, prompts, and analyses, and may use chat.
 * Writes stay with admins.
 */
const statement = {
	...defaultStatements,
	...applicationStatements,
} as const;

export const ac = createAccessControl(statement);

export const roles = {
	[Role.Admin]: ac.newRole({
		...adminAc.statements,
		...roleAccess[Role.Admin],
	}),
	[Role.Viewer]: ac.newRole({
		...roleAccess[Role.Viewer],
	}),
};

/**
 * Shape accepted by a role's `authorize()` — a partial map of resource → the
 * actions being requested, e.g. `{ analytics: ['read'] }`. Both roles share the
 * same access-control statements, so either role's request type works.
 */
export type PermissionRequest = Parameters<(typeof roles)[Role.Admin]['authorize']>[0];
