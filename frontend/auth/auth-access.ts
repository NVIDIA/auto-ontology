// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAccessControl } from 'better-auth/plugins/access';
import { adminAc, defaultStatements } from 'better-auth/plugins/admin/access';
import { Role } from '@/enums/auth';

/**
 * Access-control statements + roles shared by the Better Auth server and client.
 * `admin` and `viewer` are the only roles. On top of the built-in `user` /
 * `session` resources (`defaultStatements`) we declare GSF's own resources, then
 * assign actions per role. Every protected API/page authorizes against these
 * (see `auth/permissions.ts`), so authorization is data here rather than scattered
 * `role === 'admin'` checks.
 */
const statement = {
	...defaultStatements,
	analytics: ['read'],
	acronym: ['read', 'create', 'update', 'delete'],
	prompt: ['read', 'create', 'update', 'delete'],
	sso: ['read', 'manage'],
	conversation: ['read', 'write', 'delete'],
	chat: ['use'],
	// Custom analyses: everyone may view; only admins may add/edit/delete.
	analysis: ['read', 'manage'],
	// Data catalog (databases/schemas/tables/columns/nodes): browsing and
	// editing node metadata. Editing is allowed for viewers too.
	catalog: ['read', 'edit'],
	// Database connections (settings): everyone may view; only admins may
	// add/test/delete (they carry credentials).
	connection: ['read', 'manage'],
	// Zones (settings): everyone may view; only admins may add/edit/delete.
	zone: ['read', 'manage'],
	// Semantic compilation (settings): admin-only toggle + manual trigger.
	semanticCompilation: ['read', 'manage'],
	// Agent settings: instance-wide visualization toggle, admin-only like the
	// other settings flags. The chat pipeline resolves it server-side, so
	// viewers never need to read it.
	visualization: ['read', 'manage'],
} as const;

export const ac = createAccessControl(statement);

export const roles = {
	// Admin: full user/session management plus every GSF resource.
	[Role.Admin]: ac.newRole({
		...adminAc.statements,
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
		semanticCompilation: ['read', 'manage'],
		visualization: ['read', 'manage'],
	}),
	// Viewer: read-only on glossary/prompts, full control of their own
	// conversations, may run chat, may view custom analyses (not add/edit them),
	// and may browse + edit the data catalog. No analytics, SSO, user mgmt, and
	// no connection/zone management (admin-only). Viewers may read zones to
	// display their own accessible zones in the term single page.
	[Role.Viewer]: ac.newRole({
		// For now viewers may fully manage the glossary, custom prompts, and
		// custom analyses (create/edit/delete), so the create + edit controls on
		// those pages work without 403s.
		acronym: ['read', 'create', 'update', 'delete'],
		prompt: ['read', 'create', 'update', 'delete'],
		analysis: ['read', 'manage'],
		conversation: ['read', 'write', 'delete'],
		chat: ['use'],
		catalog: ['read', 'edit'],
		zone: ['read'],
	}),
};

/**
 * Shape accepted by a role's `authorize()` — a partial map of resource → the
 * actions being requested, e.g. `{ analytics: ['read'] }`. Both roles share the
 * same access-control statements, so either role's request type works.
 */
export type PermissionRequest = Parameters<(typeof roles)[Role.Admin]['authorize']>[0];
