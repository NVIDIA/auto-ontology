// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { userCan, type PermissionRequest } from '@/auth/permissions';
import type { ResolvedUser } from '@/auth/resolve-user';
import { Role } from '@/enums/auth';
import { ComposerSectionKind } from '@/enums/datasources';
import type { ComposerOwnerSection } from '@/types/composer-section';
import type { Owner } from '@/types/owners';

/** Section id a save reads the chosen owner from. It is the column name. */
export const OWNER_SECTION_ID = 'owner_id';

const GROUP_COLOR_CLASS: Record<string, string> = {
	'#0ea5e9': 'bg-sky-500',
	'#8b5cf6': 'bg-violet-500',
	'#ec4899': 'bg-pink-500',
	'#f97316': 'bg-orange-500',
	'#14b8a6': 'bg-teal-500',
	'#eab308': 'bg-yellow-500',
};

/** Tailwind background for a group's stored hex, or a neutral swatch. */
export const groupColorClass = (color: string | null | undefined): string =>
	(color != null && GROUP_COLOR_CLASS[color]) || 'bg-zinc-400';

export const filterOwnersByPermission = (owners: Owner[], permission: string): Owner[] =>
	owners.filter((owner) => owner.permissions.includes(permission));

export const findOwner = (
	owners: Owner[],
	ownerId: string | null | undefined,
): Owner | undefined => (ownerId ? owners.find((owner) => owner.id === ownerId) : undefined);

/** Eligible owners, with the current one kept even when they no longer qualify. */
export const ownerChoices = (
	owners: Owner[],
	permission: string,
	currentId?: string | null,
): Owner[] => {
	const eligible = filterOwnersByPermission(owners, permission);
	const current = findOwner(owners, currentId);
	if (current != null && !eligible.some((owner) => owner.id === current.id)) {
		return [current, ...eligible];
	}
	return eligible;
};

/** The signed-in account, when that account is allowed to own this kind of object. */
export const defaultOwnerId = (
	owners: Owner[],
	permission: string,
	userId: string | null | undefined,
): string =>
	userId != null &&
	filterOwnersByPermission(owners, permission).some((owner) => owner.id === userId)
		? userId
		: '';

/**
 * The owner a save should write.
 *
 * Undefined when the field was not edited. Null clears the owner. A string
 * assigns that user or group.
 */
export const changedOwnerId = (
	edit: unknown,
	current: string | null | undefined,
): string | null | undefined => {
	if (typeof edit !== 'string') return undefined;
	const next = edit.trim();
	if (next === (current ?? '')) return undefined;
	return next === '' ? null : next;
};

const ownerGrants = (owner: Owner, permission: PermissionRequest): boolean => {
	const requested = Object.entries(permission as Record<string, readonly string[]>).flatMap(
		([resource, actions]) => actions.map((action) => `${resource}:${action}`),
	);
	return requested.every((item) => owner.permissions.includes(item));
};

/**
 * Whether this account may edit through its own role or the entity's owner
 * group. Group permissions apply only when that exact group owns the entity.
 * Until the account is known this answers yes, so Edit does not flash off
 * before the session arrives. The write is checked again on the server.
 */
export const canEditEntity = ({
	user,
	permission,
	ownerId,
	owners = [],
}: {
	user: ResolvedUser | null;
	permission: PermissionRequest;
	ownerId?: string | null;
	owners?: Owner[];
}): boolean => {
	if (user == null) return true;
	if (user.role === Role.Admin) return true;
	if (userCan(user, permission)) return true;

	const owner = findOwner(owners, ownerId);
	return (
		owner?.kind === 'group' &&
		owner.member_ids.includes(user.id) &&
		ownerGrants(owner, permission)
	);
};

/** The owner card a detail page puts next to the object's other fields. */
export const entityOwnerSection = (
	ownerId: string | null | undefined,
	owners: Owner[],
	permission: string,
	editable = true,
	title = 'Owner',
): ComposerOwnerSection => ({
	type: ComposerSectionKind.OWNER,
	id: OWNER_SECTION_ID,
	title,
	ownerId: ownerId ?? null,
	permission,
	options: (editable ? ownerChoices(owners, permission, ownerId) : owners).map((owner) => ({
		id: owner.id,
		name: owner.name,
		kind: owner.kind,
		color: owner.color,
	})),
	editable,
});
