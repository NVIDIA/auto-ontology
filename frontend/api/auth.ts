// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { authClient } from '@/auth/auth-client';

export type SsoProvider = { provider_id: string; issuer: string; domain: string };

/**
 * `GET /api/sso-discovery`, which relays the OpenID Connect Discovery document
 * under its standard field names. Better Auth's `oidcConfig` spells the same
 * fields in camelCase — `register` below maps between the two.
 */
export type SsoDiscovery = {
	issuer: string;
	authorization_endpoint: string;
	token_endpoint: string;
	userinfo_endpoint: string | null;
	jwks_uri: string | null;
	discovery_endpoint: string;
};

type DiscoverResult = { discovery: SsoDiscovery; error: null } | { discovery: null; error: string };

type RegisterInput = {
	providerId: string;
	issuer: string;
	clientId: string;
	clientSecret: string;
	discovery: SsoDiscovery;
};

export const authApi = {
	/** Public list of configured providers (no secrets). */
	listProviders: async (): Promise<SsoProvider[]> => {
		const res = await fetch('/api/sso-providers').catch(() => null);
		const data = res ? await res.json().catch(() => null) : null;
		return data?.providers ?? [];
	},

	/**
	 * Resolve an issuer's OIDC endpoints via the server-side discovery helper, so
	 * the provider can be registered with `skipDiscovery` (no trustedOrigins
	 * needed for public IdPs). Returns `{ discovery }` or `{ error }`.
	 */
	discover: async (issuer: string): Promise<DiscoverResult> => {
		const res = await fetch(`/api/sso-discovery?issuer=${encodeURIComponent(issuer)}`).catch(
			() => null,
		);
		const body = res ? await res.json().catch(() => null) : null;
		if (!res || !res.ok || !body) {
			return {
				discovery: null,
				error: body?.error ?? 'Could not fetch the provider configuration from the issuer.',
			};
		}
		return { discovery: body as SsoDiscovery, error: null };
	},

	/** Register the OIDC provider (Better Auth). Domain routing is unused, so ''. */
	register: ({ providerId, issuer, clientId, clientSecret, discovery }: RegisterInput) =>
		authClient.sso.register({
			providerId,
			issuer,
			domain: '',
			oidcConfig: {
				clientId,
				clientSecret,
				skipDiscovery: true,
				authorizationEndpoint: discovery.authorization_endpoint,
				tokenEndpoint: discovery.token_endpoint,
				userInfoEndpoint: discovery.userinfo_endpoint ?? undefined,
				jwksEndpoint: discovery.jwks_uri ?? undefined,
				discoveryEndpoint: discovery.discovery_endpoint,
				scopes: ['openid', 'profile', 'email'],
				pkce: true,
			},
		}),

	deleteProvider: (providerId: string) => authClient.sso.deleteProvider({ providerId }),

	/** Email/password sign-in (the local bootstrap-admin fallback). */
	signInWithPassword: (email: string, password: string) =>
		authClient.signIn.email({ email, password }),

	/** Start the OIDC sign-in redirect for a registered provider. */
	signInWithProvider: (providerId: string, callbackURL: string) =>
		authClient.signIn.sso({ providerId, callbackURL }),
};
