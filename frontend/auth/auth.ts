// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { betterAuth } from 'better-auth';
import { prismaAdapter } from 'better-auth/adapters/prisma';
import { admin } from 'better-auth/plugins/admin';
import { nextCookies } from 'better-auth/next-js';
import { createAuthMiddleware } from 'better-auth/api';
import { sso } from '@better-auth/sso';
import { apiKey } from '@better-auth/api-key';
import { mcp } from '@better-auth/mcp';
import { getPrisma } from '@/lib/prisma';
import { ac, roles } from '@/auth/auth-access';
import { Role } from '@/enums/auth';

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

const isHttpRedirectUri = (value: unknown): value is string => {
	if (typeof value !== 'string') return false;
	try {
		const url = new URL(value);
		return url.protocol === 'http:' || url.protocol === 'https:';
	} catch {
		return false;
	}
};

const prisma = getPrisma();

// During `next build` the module is evaluated but no auth request is handled,
// so a real secret/origin isn't needed. Fall back to a build-only placeholder
// for the secret so construction doesn't fail; runtime values come from env.
const isBuildPhase = process.env.NEXT_PHASE === 'phase-production-build';

const mcpResource = (): string => {
	const publicUrl = process.env.AUTO_ONTOLOGY_MCP_PUBLIC_URL?.replace(/\/+$/, '');
	if (publicUrl) return `${publicUrl}/mcp`;
	if (isBuildPhase) return 'https://mcp.invalid/mcp';
	return 'http://localhost:3003/mcp';
};

/** True once at least one SSO provider has been registered. */
export const isSsoConfigured = async (): Promise<boolean> => (await prisma.ssoProvider.count()) > 0;

export const auth = betterAuth({
	// Project-scoped env var names, wired explicitly so they aren't tied to
	// Better Auth's BETTER_AUTH_* defaults.
	secret: process.env.AUTH_SECRET ?? (isBuildPhase ? 'next-build-time-placeholder' : undefined),
	baseURL: process.env.APP_URL,
	// Cursor's desktop callback is a custom-scheme URI. Without this, the
	// authorize/consent hop can fail origin checks after DCR succeeds.
	trustedOrigins: [
		process.env.APP_URL,
		'cursor://',
		'http://localhost:8787',
		'http://127.0.0.1:8787',
		'https://www.cursor.com',
	].filter((origin): origin is string => Boolean(origin)),
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
	// AUTO_ONTOLOGY_ADMIN_EMAIL / AUTO_ONTOLOGY_ADMIN_PASSWORD (see lib/seed-admin.ts). Further users
	// are added by an admin or provisioned via SSO.
	emailAndPassword: { enabled: true, disableSignUp: true },
	// MCP clients (Cursor, the Python SDK) register loopback HTTP and
	// `cursor://` redirect URIs. Better Auth 1.7 defaults an omitted
	// `application_type` to `web`, and Cursor sometimes sends `web` itself —
	// either way the web-client rule rejects those URIs. This authorization
	// server exists for MCP, so DCR is always native.
	//
	// Unpatched Better Auth also rejects Cursor's host-bearing `cursor://`
	// callback (better-auth#10946). Cursor still registers loopback
	// `http://localhost:8787/callback` and `https://www.cursor.com/...`, which
	// native DCR accepts, so drop the custom-scheme extras rather than patch
	// the package.
	hooks: {
		before: createAuthMiddleware(async (ctx) => {
			if (ctx.path !== '/oauth2/register') return;
			const body = ctx.body;
			if (!isPlainObject(body)) return;
			return {
				context: {
					body: {
						...body,
						application_type: 'native',
						redirect_uris: Array.isArray(body.redirect_uris)
							? body.redirect_uris.filter(isHttpRedirectUri)
							: body.redirect_uris,
					},
				},
			};
		}),
	},
	plugins: [
		// Two roles only: `admin` (user management) and `viewer` (everything else).
		admin({ ac, roles, adminRoles: [Role.Admin], defaultRole: Role.Viewer }),
		// Auto Ontology Allows only one SSO provider, so this makes the redirect URI static.
		// Better Auth's baseURL already includes `/api/auth`, so this becomes
		// APP_URL/api/auth/sso/callback — the URI the README tells the IdP to register.
		sso({ redirectURI: '/sso/callback' }),
		// Machine-to-machine credentials: long-lived API tokens users mint for
		// scripts. The plugin owns hashing (SHA-256), expiry, and revocation; we
		// only pin the ergonomics. Tokens are verified explicitly in
		// auth/api-token.ts rather than via `enableSessionForAPIKeys` (left off,
		// its default), so a token never silently becomes a browser session.
		apiKey({
			// `auto_ontology_` makes a leaked token greppable in logs and recognisable to
			// secret scanners; `start` keeps the first few characters so the UI can
			// identify a token it can no longer read.
			defaultPrefix: 'auto_ontology_',
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
		// Makes Auto Ontology the authorization server that MCP clients log in against, so
		// the MCP server holds no client credentials and NVIDIA never needs a
		// redirect URI registered for it — the only one it ever sees is Auto Ontology's,
		// which SSO already uses. Clients register themselves, and those
		// registrations live in Postgres rather than on an MCP pod's disk.
		//
		// `loginPage` is where an unauthenticated authorize request is sent; from
		// there the existing SSO flow takes over, so human login is unchanged.
		mcp({
			loginPage: '/login',
			consentPage: '/oauth/consent',
			resource: mcpResource(),
			// UserInfo requires `openid`. Advertise it so MCP clients that copy
			// `scopes_supported` from authorization-server metadata include it.
			scopes: ['openid', 'profile', 'email', 'offline_access'],
			// The MCP server checks every token against Auto Ontology, so opaque tokens retain
			// immediate revocation without distributing resource-server credentials.
			disableJwtPlugin: true,
			// Keep RFC 7591 registration for existing MCP clients. Both switches are
			// required in 1.7; registration is no longer enabled by `mcp()` itself.
			allowDynamicClientRegistration: true,
			allowUnauthenticatedClientRegistration: true,
			// UserInfo is the bearer-authenticated replacement for getMcpSession.
			// Include the grant metadata FastMCP needs under private claim names.
			customUserInfoClaims: ({ jwt }) => ({
				'urn:auto-ontology:oauth:client_id': jwt.client_id,
				'urn:auto-ontology:oauth:scope': jwt.scope,
			}),
			// Every MCP call validates its token here. The app-wide limiter still
			// applies; the provider's 60/minute UserInfo limit is too low for agents.
			rateLimit: { userinfo: false },
		}),
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
