// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Icon, IconName } from '@/components/icons';
import { authApi, type SsoProvider } from '@/api/auth';
import { useSession } from '@/auth/auth-client';

const inputClass =
	'w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300';

const LoginForm = () => {
	const router = useRouter();
	const params = useSearchParams();
	const next = params.get('next') || '/chat';

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
		if (session) {
			router.replace(next);
		}
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
		router.push(next);
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
		return <p className="w-full max-w-sm text-center text-sm text-zinc-500" />;
	}

	// SSO configured: show a "Sign in with SSO" button (no auto-login) so the
	// user explicitly starts the flow, plus the password backdoor for the local
	// bootstrap admin.
	if (ssoEnabled && !passwordMode) {
		return (
			<div className="flex w-full max-w-sm flex-col gap-4">
				{providers.map((provider) => (
					<button
						key={provider.providerId}
						type="button"
						onClick={() => handleSso(provider.providerId)}
						className="cursor-pointer rounded-md bg-[#76b900] px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#6aa600]"
					>
						Sign in with SSO
					</button>
				))}
				<button
					type="button"
					onClick={() => {
						setError(null);
						setShowPasswordLogin(true);
					}}
					className="cursor-pointer text-center text-xs text-zinc-400 transition-colors hover:text-zinc-600 hover:underline dark:hover:text-zinc-300"
				>
					Sign in with password
				</button>
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

			<button
				type="submit"
				disabled={submitting}
				className="cursor-pointer rounded-md bg-[#76b900] px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#6aa600] disabled:opacity-50"
			>
				{submitting ? 'Signing in…' : 'Sign in'}
			</button>

			{ssoEnabled ? (
				<button
					type="button"
					onClick={() => {
						setError(null);
						setShowPasswordLogin(false);
					}}
					className="cursor-pointer text-center text-xs text-zinc-400 transition-colors hover:text-zinc-600 hover:underline dark:hover:text-zinc-300"
				>
					Back to SSO
				</button>
			) : null}
		</form>
	);
};

const LoginPage = () => (
	<div className="flex h-full w-full items-center justify-center bg-white p-6 dark:bg-zinc-950">
		<div className="flex flex-col items-center gap-6">
			<div className="flex items-center gap-2">
				<Icon name={IconName.NvidiaLogo} className="h-6 w-6" />
				<span className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">GSF</span>
			</div>
			<Suspense fallback={null}>
				<LoginForm />
			</Suspense>
		</div>
	</div>
);

export default LoginPage;
