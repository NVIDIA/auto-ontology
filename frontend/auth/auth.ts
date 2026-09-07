// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { betterAuth } from 'better-auth';
import { prismaAdapter } from 'better-auth/adapters/prisma';
import { admin } from 'better-auth/plugins/admin';
// Imported from the barrel, not `better-auth/plugins/mcp`: the package exports
// only the plugin's client subpath, so the server plugin is reachable here.
import { mcp } from 'better-auth/plugins';
import { nextCookies } from 'better-auth/next-js';
import { sso } from '@better-auth/sso';
import { apiKey } from '@better-auth/api-key';
import { getPrisma } from '@/lib/prisma';
import { ac, roles } from '@/auth/auth-access';
import { Role } from '@/enums/auth';

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
	// Better Auth caps any /sign-in* path at 3 requests per 10s by default, which
	// rejects legitimate bursts: several parallel sessions for one account, or many
	// users behind one egress IP (the bucket is keyed by IP + path). Raise just that
	// path; the global default (100/10s) still applies everywhere else.
	rateLimit: {
		customRules: {
			'/sign-in/*': { window: 10, max: 20 },
		},
	},
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
		// Machine-to-machine credentials: long-lived API tokens users mint for
		// scripts. The plugin owns hashing (SHA-256), expiry, and revocation; we
		// only pin the ergonomics. Tokens are verified explicitly in
		// auth/api-token.ts rather than via `enableSessionForAPIKeys` (left off,
		// its default), so a token never silently becomes a browser session.
		apiKey({
			// `gsf_` makes a leaked token greppable in logs and recognisable to
			// secret scanners; `start` keeps the first few characters so the UI can
			// identify a token it can no longer read.
			defaultPrefix: 'gsf_',
			defaultKeyLength: 48,
			startingCharactersConfig: { shouldStore: true, charactersLength: 10 },
			// A token without a name is unidentifiable in the revoke list.
			requireName: true,
			minimumNameLength: 1,
			maximumNameLength: 64,
			// The plugin's default is 10 requests per day, which would break any
			// real script. Throttling is the app-wide rateLimit's job, not the
			// token's.
			rateLimit: { enabled: false },
			// Non-expiring by default (a script's credential should not silently
			// die), but callers may opt into an expiry up to a year out.
			keyExpiration: { defaultExpiresIn: null, maxExpiresIn: 365 },
		}),
		// Makes GSF the authorization server that MCP clients log in against, so
		// the MCP server holds no client credentials and NVIDIA never needs a
		// redirect URI registered for it — the only one it ever sees is GSF's,
		// which SSO already uses. Clients register themselves, and those
		// registrations live in Postgres rather than on an MCP pod's disk.
		//
		// `loginPage` is where an unauthenticated authorize request is sent; from
		// there the existing SSO flow takes over, so human login is unchanged.
		mcp({ loginPage: '/login' }),
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
			},
		},
	},
});
