// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAccessControl } from 'better-auth/plugins/access';
import { adminAc, defaultStatements } from 'better-auth/plugins/admin/access';
import { Role } from '@/enums/auth';

/**
 * Access-control statements + roles shared by the Better Auth server and client.
 * `admin` and `viewer` are the only roles. On top of the built-in `user` /
 * `session` resources (`defaultStatements`) we declare Auto Ontology's own resources, then
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
	// Tags: `manage` is the vocabulary — creating and deleting tags on the
	// settings page — and stays admin-only. `read` goes to viewers too, because
	// the term page offers every existing tag in its picker, and applying one is
	// guarded by `catalog: ['edit']` instead: curating the list of tags and
	// labelling an object with one already on it are different privileges.
	tag: ['read', 'manage'],
	// Semantic compilation (settings): admin-only toggle + manual trigger.
	semanticCompilation: ['read', 'manage'],
	// Agent settings: the instance-wide visualization toggle and SQL query
	// timeout, admin-only like the other settings flags. The chat pipeline
	// resolves both server-side, so viewers never need to read them.
	visualization: ['read', 'manage'],
	// Model import/export (settings): admin-only YAML backup/restore of the
	// catalog + semantic layer.
	modelInterchange: ['export', 'import'],
	// API tokens for scripting. Self-service for every role — the routes scope
	// each call to the caller's own tokens, and a token can never exceed the
	// permissions its owner already has.
	apiToken: ['manage'],
} as const;

export const ac = createAccessControl(statement);

export const roles = {
	// Admin: full user/session management plus every Auto Ontology resource.
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
		tag: ['read', 'manage'],
		semanticCompilation: ['read', 'manage'],
		visualization: ['read', 'manage'],
		modelInterchange: ['export', 'import'],
		apiToken: ['manage'],
	}),
	// Viewer: read-only on glossary/prompts, full control of their own
	// conversations, may run chat, may view custom analyses (not add/edit them),
	// and may browse + edit the data catalog. No analytics, SSO, user mgmt, and
	// no connection/zone management (admin-only). Viewers may read zones and
	// tags to display them in the term single page, and label objects with an
	// existing tag under the catalog edit permission they already hold. They may
	// also read connections to see which databases are wired up — the list route
	// strips credentials, so only the shape of a connection is exposed.
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
		connection: ['read'],
		zone: ['read'],
		tag: ['read'],
		apiToken: ['manage'],
	}),
};

/**
 * Shape accepted by a role's `authorize()` — a partial map of resource → the
 * actions being requested, e.g. `{ analytics: ['read'] }`. Both roles share the
 * same access-control statements, so either role's request type works.
 */
export type PermissionRequest = Parameters<(typeof roles)[Role.Admin]['authorize']>[0];
