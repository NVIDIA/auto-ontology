// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from 'node:crypto';
import { hashPassword } from 'better-auth/crypto';
import { getPrisma } from '@/lib/prisma';
import { Role } from '@/enums/auth';

/**
 * Idempotently ensure the bootstrap admin account described by the
 * AUTO_ONTOLOGY_ADMIN_EMAIL / AUTO_ONTOLOGY_ADMIN_PASSWORD env vars. Runs once at server startup
 * (see `instrumentation.ts`).
 *
 * - If no account with that email exists, it is created as an email-verified
 *   admin with a `credential` (email/password) account.
 * - If it already exists, its password is reset to the configured value and
 *   its admin role + verified status are re-asserted.
 *
 * Self-service sign-up is disabled. Other credential accounts are created
 * when an invite is accepted. No-op unless BOTH env vars are set.
 *
 * The password is hashed with Better Auth's own `hashPassword`, so the stored
 * value verifies against the normal email/password sign-in flow.
 */
export const seedAdmin = async (): Promise<void> => {
	const email = process.env.AUTO_ONTOLOGY_ADMIN_EMAIL?.trim().toLowerCase();
	const password = process.env.AUTO_ONTOLOGY_ADMIN_PASSWORD;
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
				name: email,
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
	console.log(`[seed-admin] updated bootstrap admin account for ${email}`);
};
