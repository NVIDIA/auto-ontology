// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from 'node:crypto';
import { hashPassword } from 'better-auth/crypto';
import { getPrisma } from '@/lib/prisma';
import { Role } from '@/enums/auth';

const pythonApiUrl = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

async function syncToNeo4j(user: {
	id: string;
	email: string;
	name: string;
	role: string;
}): Promise<void> {
	try {
		await fetch(`${pythonApiUrl}/api/users`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(user),
		});
	} catch (err) {
		console.error('[seed-admin] failed to sync admin to Neo4j', err);
	}
}

/**
 * Idempotently ensure the bootstrap admin account described by the
 * GSF_ADMIN_EMAIL / GSF_ADMIN_PASSWORD env vars. Runs once at server startup
 * (see `instrumentation.ts`).
 *
 * - If no account with that email exists, it is created as an email-verified
 *   admin with a `credential` (email/password) account.
 * - If it already exists, its password is reset to the configured value and
 *   its admin role + verified status are re-asserted.
 *
 * Self-service sign-up is disabled, so this is the only way a credential
 * account is created. No-op unless BOTH env vars are set.
 *
 * The password is hashed with Better Auth's own `hashPassword`, so the stored
 * value verifies against the normal email/password sign-in flow.
 */
export const seedAdmin = async (): Promise<void> => {
	const email = process.env.GSF_ADMIN_EMAIL?.trim().toLowerCase();
	const password = process.env.GSF_ADMIN_PASSWORD;
	if (!email || !password) return;

	const prisma = getPrisma();
	const hashedPassword = await hashPassword(password);

	const existing = await prisma.user.findFirst({ where: { email } });

	if (!existing) {
		const userId = randomUUID();
		await prisma.user.create({
			data: {
				id: userId,
				email,
				name: 'Admin',
				emailVerified: true,
				role: Role.Admin,
			},
		});
		await prisma.account.create({
			data: {
				id: randomUUID(),
				accountId: userId,
				providerId: 'credential',
				userId,
				password: hashedPassword,
			},
		});
		await syncToNeo4j({ id: userId, email, name: 'Admin', role: Role.Admin });
		console.log(`[seed-admin] created bootstrap admin account for ${email}`);
		return;
	}

	await prisma.user.update({
		where: { id: existing.id },
		data: { role: Role.Admin, emailVerified: true },
	});

	const credential = await prisma.account.findFirst({
		where: { userId: existing.id, providerId: 'credential' },
	});
	if (credential) {
		await prisma.account.update({
			where: { id: credential.id },
			data: { password: hashedPassword },
		});
	} else {
		await prisma.account.create({
			data: {
				id: randomUUID(),
				accountId: existing.id,
				providerId: 'credential',
				userId: existing.id,
				password: hashedPassword,
			},
		});
	}
	await syncToNeo4j({ id: existing.id, email, name: existing.name, role: Role.Admin });
	console.log(`[seed-admin] updated bootstrap admin account for ${email}`);
};
