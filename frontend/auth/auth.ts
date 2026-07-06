// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { betterAuth } from 'better-auth';
import { prismaAdapter } from 'better-auth/adapters/prisma';
import { admin } from 'better-auth/plugins/admin';
import { nextCookies } from 'better-auth/next-js';
import { sso } from '@better-auth/sso';
import { getPrisma } from '@/lib/prisma';
import { ac, roles } from '@/auth/auth-access';
import { Role } from '@/enums/auth';

const pythonApiUrl = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

async function syncUserToNeo4j(user: {
	id: string;
	email: string;
	name: string;
	role?: string | null;
}): Promise<void> {
	try {
		await fetch(`${pythonApiUrl}/api/users`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				id: user.id,
				email: user.email,
				name: user.name,
				role: user.role ?? Role.Viewer,
			}),
		});
	} catch (err) {
		console.error('[auth] failed to sync user to Neo4j', err);
	}
}

const prisma = getPrisma();

// During `next build` the module is evaluated but no auth request is handled,
// so a real secret/origin isn't needed. Fall back to a build-only placeholder
// for the secret so construction doesn't fail; runtime values come from env.
const isBuildPhase = process.env.NEXT_PHASE === 'phase-production-build';

/** True once at least one SSO provider has been registered. */
export const isSsoConfigured = async (): Promise<boolean> => (await prisma.ssoProvider.count()) > 0;

export const auth = betterAuth({
	// Project-scoped env var names, wired explicitly so they aren't tied to
	// Better Auth's BETTER_AUTH_* defaults.
	secret: process.env.AUTH_SECRET ?? (isBuildPhase ? 'next-build-time-placeholder' : undefined),
	baseURL: process.env.APP_URL,
	database: prismaAdapter(prisma, { provider: 'postgresql' }),
	// Email/password sign-IN is enabled, but self-service sign-UP is disabled:
	// the only credential account is the bootstrap admin seeded from
	// GSF_ADMIN_EMAIL / GSF_ADMIN_PASSWORD (see lib/seed-admin.ts). Further users
	// are added by an admin or provisioned via SSO.
	emailAndPassword: { enabled: true, disableSignUp: true },
	plugins: [
		// Two roles only: `admin` (user management) and `viewer` (everything else).
		admin({ ac, roles, adminRoles: [Role.Admin], defaultRole: Role.Viewer }),
		// GSF Allows only one SSO provider, so this makes the redirect URI static.
		// This resolves to /api/auth/sso/callback
		// the `ssoProvider` table — there are no SSO env vars.
		sso({ redirectURI: '/sso/callback' }),
		// Must be the last plugin so it can set cookies on outgoing responses.
		nextCookies(),
	],
	databaseHooks: {
		user: {
			create: {
				// First account ever created becomes the admin and is marked
				// email-verified (so it can later link an SSO identity to the same
				// account); everyone else is a viewer. The unique email constraint
				// guards against a duplicate-account race; the count check assigns
				// the role.
				before: async (user) => {
					const isFirstUser = (await prisma.user.count()) === 0;
					const name = user.name || user.email;
					return {
						data: isFirstUser
							? { ...user, name, role: Role.Admin, emailVerified: true }
							: { ...user, name, role: Role.Viewer },
					};
				},
				after: async (user) => {
					await syncUserToNeo4j(user);
				},
			},
			update: {
				after: async (user) => {
					// Sync role / name changes (e.g. admin promotes a viewer).
					await syncUserToNeo4j(user);
				},
			},
		},
	},
});
