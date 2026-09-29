// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { Icon, IconName } from '@/common/icons';
import { SkeletonBlock } from '@/common/Skeleton';
import { authApi, type SsoProvider } from '@/api/auth';
import { useSession } from '@/auth/auth-client';
import { resumeAuthorizeFromApp } from '@/auth/oauth-loopback';
import { SkeletonVariant } from '@/enums/skeleton';

const inputClass =
	'w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300';

const LoginLoading = () => (
	<div
		className="w-full max-w-sm space-y-4 py-6"
		role="status"
		aria-label="Loading sign-in options"
	>
		<SkeletonBlock variant={SkeletonVariant.RECTANGLE} className="h-10 w-full" />
		<SkeletonBlock variant={SkeletonVariant.RECTANGLE} className="h-10 w-full" />
		<SkeletonBlock className="mx-auto h-3 w-1/3" />
	</div>
);

// An MCP client sends a user here to sign in, by way of an authorize request
// that arrived without a session (see the `mcp` plugin's `loginPage` in
// auth/auth.ts). Its OAuth parameters come along on the query string, and
// signing in has to hand the user back to the authorize endpoint with them —
// otherwise the browser lands on /chat and the client that started the flow
// waits for a code that will never come.
const AUTHORIZE_PATH = '/api/auth/oauth2/authorize';

const resumeAuthorize = (params: URLSearchParams): string | null =>
	params.has('client_id') && params.get('response_type') === 'code'
		? `${AUTHORIZE_PATH}?${params.toString()}`
		: null;

const LoginForm = () => {
	const router = useRouter();
	const params = useSearchParams();
	const next = params.get('next') || resumeAuthorize(params) || '/chat';

	// The authorize endpoint is a route handler, not a page. Resume it with a
	// same-origin fetch so a loopback redirect_uri stays on Auto Ontology's thank-you
	// page instead of Chrome following localhost:8787.
	const goTo = (destination: string) => {
		if (destination.startsWith('/api/')) {
			void resumeAuthorizeFromApp(destination);
			return;
		}
		router.push(destination);
	};

	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [error, setError] = useState<string | null>(null);
	const [submitting, setSubmitting] = useState(false);
	const [providers, setProviders] = useState<SsoProvider[]>([]);
	const [providersLoaded, setProvidersLoaded] = useState(false);
	// When SSO is configured, SSO is the only option shown by default. This
	// toggles the email/password fallback ("backdoor") so the local bootstrap
	// admin can still sign in if the SSO provider is unreachable.
	const [showPasswordLogin, setShowPasswordLogin] = useState(false);

	// Authoritative session check (validates the cookie server-side and clears
	// it if invalid). Used to send an already-authenticated user to their
	// destination — the middleware no longer bounces cookie-bearing requests off
	// /login, since a stale cookie there would loop against requireUser().
	const { data: session, isPending: sessionPending } = useSession();

	useEffect(() => {
		authApi
			.listProviders()
			.then(setProviders)
			.finally(() => setProvidersLoaded(true));
	}, []);

	useEffect(() => {
		if (!session) return;
		if (next.startsWith('/api/')) {
			void resumeAuthorizeFromApp(next);
			return;
		}
		router.replace(next);
	}, [session, router, next]);

	const ssoEnabled = providers.length > 0;

	const handleSubmit = async (event: React.FormEvent) => {
		event.preventDefault();
		setSubmitting(true);
		setError(null);
		const result = await authApi.signInWithPassword(email, password);
		if (result.error) {
			setError(result.error.message ?? 'Invalid email or password.');
			setSubmitting(false);
			return;
		}
		goTo(next);
		router.refresh();
	};

	const handleSso = (providerId: string) => {
		authApi.signInWithProvider(providerId, next);
	};

	// The password "backdoor" — reachable via /login?password (or the link
	// below) so the local admin can still sign in if SSO is down.
	const passwordMode = showPasswordLogin || params.has('password');

	// Avoid flashing the form before we know whether SSO is configured, and while
	// an already-authenticated user is being redirected to their destination.
	if (!providersLoaded || sessionPending || session) {
		return <LoginLoading />;
	}

	// SSO configured: show a "Sign in with SSO" button (no auto-login) so the
	// user explicitly starts the flow, plus the password backdoor for the local
	// bootstrap admin.
	if (ssoEnabled && !passwordMode) {
		return (
			<div className="flex w-full max-w-sm flex-col gap-4">
				{providers.map((provider) => (
					<Button
						theme={ButtonTheme.Primary}
						size={Size.REGULAR}
						key={provider.provider_id}
						type="button"
						onClick={() => handleSso(provider.provider_id)}
					>
						Sign in with SSO
					</Button>
				))}
				<Button
					theme={ButtonTheme.Minimal}
					size={Size.SMALL}
					type="button"
					onClick={() => {
						setError(null);
						setShowPasswordLogin(true);
					}}
				>
					Sign in with password
				</Button>
			</div>
		);
	}

	// No SSO configured, or the admin chose the password fallback.
	return (
		<form onSubmit={handleSubmit} className="flex w-full max-w-sm flex-col gap-4">
			<div className="flex flex-col gap-1.5">
				<label
					htmlFor="email"
					className="text-xs font-medium text-zinc-600 dark:text-zinc-400"
				>
					Email
				</label>
				<input
					id="email"
					type="email"
					autoComplete="email"
					required
					value={email}
					onChange={(event) => setEmail(event.target.value)}
					className={inputClass}
				/>
			</div>
			<div className="flex flex-col gap-1.5">
				<label
					htmlFor="password"
					className="text-xs font-medium text-zinc-600 dark:text-zinc-400"
				>
					Password
				</label>
				<input
					id="password"
					type="password"
					autoComplete="current-password"
					required
					value={password}
					onChange={(event) => setPassword(event.target.value)}
					className={inputClass}
				/>
			</div>

			{error ? <p className="text-xs text-red-500">{error}</p> : null}

			<Button
				theme={ButtonTheme.Primary}
				size={Size.REGULAR}
				type="submit"
				disabled={submitting}
			>
				{submitting ? 'Signing in…' : 'Sign in'}
			</Button>

			{ssoEnabled ? (
				<Button
					theme={ButtonTheme.Minimal}
					size={Size.SMALL}
					type="button"
					onClick={() => {
						setError(null);
						setShowPasswordLogin(false);
					}}
				>
					Back to SSO
				</Button>
			) : null}
		</form>
	);
};

const LoginPage = () => (
	<div className="flex h-full w-full items-center justify-center bg-white p-6 dark:bg-zinc-950">
		<div className="flex flex-col items-center gap-6">
			<div className="flex items-center gap-2">
				<Icon name={IconName.NvidiaLogo} className="h-6 w-6" />
				<span className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
					Auto Ontology
				</span>
			</div>
			<Suspense fallback={<LoginLoading />}>
				<LoginForm />
			</Suspense>
		</div>
	</div>
);

export default LoginPage;
