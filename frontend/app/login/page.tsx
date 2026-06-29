// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Icon, IconName } from '@/components/icons';
import { authApi, type SsoProvider } from '@/api/auth';

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

	useEffect(() => {
		authApi
			.listProviders()
			.then(setProviders)
			.finally(() => setProvidersLoaded(true));
	}, []);

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
	// A failed SSO callback bounces back to /login carrying an error; don't
	// auto-redirect in that case or we'd loop forever back to the IdP.
	const hasError = params.has('error') || next.includes('error=');
	// When SSO is configured, send the user straight to the provider on load
	// (no button click) — unless they want the password form or just errored out.
	const autoRedirect = ssoEnabled && !passwordMode && !hasError;

	useEffect(() => {
		if (providersLoaded && autoRedirect && providers[0]) {
			authApi.signInWithProvider(providers[0].providerId, next);
		}
	}, [providersLoaded, autoRedirect, providers, next]);

	// Avoid flashing the form before providers load, and while the auto-redirect
	// to the IdP is kicking off.
	if (!providersLoaded || autoRedirect) {
		return (
			<p className="w-full max-w-sm text-center text-sm text-zinc-500">
				{autoRedirect ? 'Redirecting to SSO…' : null}
			</p>
		);
	}

	// SSO configured but auto-redirect was suppressed (an error bounced the user
	// back): offer the SSO button to retry, plus the password backdoor.
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
