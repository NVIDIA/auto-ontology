// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash, randomBytes, randomUUID } from 'node:crypto';
import { hashPassword } from 'better-auth/crypto';
import { Role } from '@/enums/auth';
import { InvitationStatus } from '@/enums/invitation';
import { getPrisma } from '@/lib/prisma';
import type { Invitation } from '@/types/invitation';

export const INVITE_TTL_MS = 48 * 60 * 60 * 1000;
export const MIN_PASSWORD_LENGTH = 8;
export const MAX_PASSWORD_LENGTH = 128;

export type OpenInvitation = {
	email: string;
	name: string;
	expires_at: Date;
};

const hashInviteToken = (token: string): string => createHash('sha256').update(token).digest('hex');

const generateInviteToken = (): string => randomBytes(32).toString('base64url');

const inviteUrl = (appUrl: string, token: string): string =>
	`${appUrl.replace(/\/+$/, '')}/invite/${token}`;

const isLiveInvitation = (row: { expires_at: Date }): boolean =>
	row.expires_at.getTime() > Date.now();

export const invitationStatus = (row: { expires_at: Date }): InvitationStatus => {
	if (row.expires_at.getTime() <= Date.now()) return InvitationStatus.Expired;
	return InvitationStatus.Active;
};

export const parseInviteRole = (value: unknown): Role | null => {
	if (value === Role.Admin || value === Role.Viewer) return value;
	return null;
};

export const getOpenInvitation = async (token: string): Promise<OpenInvitation | null> => {
	if (!token) return null;
	const row = await getPrisma().invitation.findUnique({
		where: { token_hash: hashInviteToken(token) },
		select: { email: true, name: true, expires_at: true },
	});
	if (!row || !isLiveInvitation(row)) return null;
	return { email: row.email, name: row.name, expires_at: row.expires_at };
};

export const listInvitations = async (appUrl: string): Promise<Invitation[]> => {
	const rows = await getPrisma().invitation.findMany({
		orderBy: { created_at: 'desc' },
	});
	const origin = appUrl.replace(/\/+$/, '');
	return rows.map((row) => {
		const status = invitationStatus(row);
		return {
			id: row.id,
			email: row.email,
			name: row.name,
			role: row.role === Role.Admin ? Role.Admin : Role.Viewer,
			status,
			expires_at: row.expires_at.toISOString(),
			created_at: row.created_at.toISOString(),
			url: status === InvitationStatus.Active ? inviteUrl(origin, row.token) : null,
		};
	});
};

export const createInvitation = async ({
	email,
	name,
	role,
	createdById,
	appUrl,
}: {
	email: string;
	name: string;
	role: Role;
	createdById: string;
	appUrl: string;
}): Promise<{ url: string; email: string; expires_at: Date }> => {
	const prisma = getPrisma();
	const existingUser = await prisma.user.findFirst({ where: { email } });
	if (existingUser) {
		throw new InviteConflictError('A user with this email already exists.');
	}

	const token = generateInviteToken();
	const expiresAt = new Date(Date.now() + INVITE_TTL_MS);

	await prisma.$transaction([
		prisma.invitation.deleteMany({ where: { email } }),
		prisma.invitation.create({
			data: {
				id: randomUUID(),
				email,
				name,
				role,
				token,
				token_hash: hashInviteToken(token),
				expires_at: expiresAt,
				created_by_id: createdById,
			},
		}),
	]);

	return { url: inviteUrl(appUrl, token), email, expires_at: expiresAt };
};

export const deleteInvitation = async (id: string): Promise<void> => {
	const prisma = getPrisma();
	const existing = await prisma.invitation.findUnique({ where: { id } });
	if (!existing) {
		throw new InviteNotFoundError();
	}
	await prisma.invitation.delete({ where: { id } });
};

export const acceptInvitation = async (
	token: string,
	password: string,
): Promise<{ email: string } | { error: string; status: number }> => {
	if (password.length < MIN_PASSWORD_LENGTH) {
		return {
			error: `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`,
			status: 400,
		};
	}
	if (password.length > MAX_PASSWORD_LENGTH) {
		return {
			error: `Password must be at most ${MAX_PASSWORD_LENGTH} characters.`,
			status: 400,
		};
	}

	const prisma = getPrisma();
	const tokenHash = hashInviteToken(token);
	const hashedPassword = await hashPassword(password);

	try {
		const email = await prisma.$transaction(async (tx) => {
			const invitation = await tx.invitation.findUnique({
				where: { token_hash: tokenHash },
			});
			if (!invitation || !isLiveInvitation(invitation)) {
				throw new InviteNotFoundError();
			}

			const existingUser = await tx.user.findFirst({
				where: { email: invitation.email },
			});
			if (existingUser) {
				throw new InviteConflictError('A user with this email already exists.');
			}

			const userId = randomUUID();
			await tx.user.create({
				data: {
					id: userId,
					email: invitation.email,
					name: invitation.name || invitation.email,
					emailVerified: true,
					role: invitation.role,
				},
			});
			await tx.account.create({
				data: {
					id: randomUUID(),
					accountId: userId,
					providerId: 'credential',
					userId,
					password: hashedPassword,
				},
			});
			await tx.invitation.delete({
				where: { id: invitation.id },
			});
			return invitation.email;
		});
		return { email };
	} catch (error) {
		if (error instanceof InviteNotFoundError) {
			return { error: 'This invitation is invalid or has expired.', status: 404 };
		}
		if (error instanceof InviteConflictError) {
			return { error: error.message, status: 409 };
		}
		throw error;
	}
};

export class InviteConflictError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'InviteConflictError';
	}
}

export class InviteNotFoundError extends Error {
	constructor() {
		super('Invitation not found');
		this.name = 'InviteNotFoundError';
	}
}
