// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { authApi, type SsoProvider } from '@/api/auth';
import { Toast } from '@/common/Toast';

// Only one provider is supported; its id is a fixed constant. It's the DB key
// and the providerId segment of the native callback (/api/auth/sso/callback/sso).
const SSO_PROVIDER_ID = 'sso';

const inputClass =
	'w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-zinc-500 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:disabled:bg-zinc-800/50 dark:disabled:text-zinc-400';

export const SsoConfigForm = ({ initialProviders }: { initialProviders: SsoProvider[] }) => {
	// Seeded from the server (see page.tsx) so the correct view renders on first
	// paint; re-fetched after register/delete to stay in sync.
	const [providers, setProviders] = useState<SsoProvider[]>(initialProviders);
	const [issuer, setIssuer] = useState('');
	const [clientId, setClientId] = useState('');
	const [clientSecret, setClientSecret] = useState('');
	const [error, setError] = useState<string | null>(null);
	const [message, setMessage] = useState<string | null>(null);
	const [submitting, setSubmitting] = useState(false);
	const [deletingId, setDeletingId] = useState<string | null>(null);

	const handleSubmit = async (event: React.FormEvent) => {
		event.preventDefault();
		setSubmitting(true);
		setError(null);
		setMessage(null);

		// Resolve the provider's endpoints server-side, then register with
		// skipDiscovery (see api/auth.ts).
		const discovered = await authApi.discover(issuer);
		if (discovered.error !== null) {
			setError(discovered.error);
			setSubmitting(false);
			return;
		}

		const result = await authApi.register({
			providerId: SSO_PROVIDER_ID,
			issuer,
			clientId,
			clientSecret,
			discovery: discovered.discovery,
		});

		if (result.error) {
			setError(result.error.message ?? 'Failed to save the SSO provider.');
			setSubmitting(false);
			return;
		}

		setMessage('SSO provider saved.');
		setClientSecret('');
		setSubmitting(false);
		setProviders(await authApi.listProviders());
	};

	const handleDelete = async (id: string) => {
		if (
			!window.confirm(
				`Delete SSO provider "${id}"? Users will no longer be able to sign in with it.`,
			)
		) {
			return;
		}

		setError(null);
		setMessage(null);
		setDeletingId(id);

		const result = await authApi.deleteProvider(id);

		if (result.error) {
			setError(result.error.message ?? 'Failed to delete the SSO provider.');
			setDeletingId(null);
			return;
		}

		setMessage(`SSO provider "${id}" deleted.`);
		setDeletingId(null);
		setProviders(await authApi.listProviders());
	};

	const provider = providers[0];
	const isConfigured = Boolean(provider);

	return (
		<div className="h-full overflow-auto p-6">
			<div className="mx-auto max-w-xl">
				<h1 className="mb-1 text-lg font-semibold text-zinc-900 dark:text-zinc-100">
					Single Sign-On (OIDC)
				</h1>
				<p className="mb-4 text-xs text-zinc-500">
					Configure a single OpenID Connect provider. Endpoints are discovered from the
					issuer. The client secret is stored securely and never shown again.
				</p>

				<form onSubmit={handleSubmit} className="flex flex-col gap-4">
					<div className="flex flex-col gap-1.5">
						<label
							htmlFor="issuer"
							className="text-xs font-medium text-zinc-600 dark:text-zinc-400"
						>
							Issuer URL
						</label>
						<input
							id="issuer"
							type="url"
							required
							placeholder="https://example.okta.com"
							disabled={isConfigured}
							value={provider ? provider.issuer : issuer}
							onChange={(event) => setIssuer(event.target.value)}
							className={inputClass}
						/>
					</div>
					<div className="flex flex-col gap-1.5">
						<label
							htmlFor="clientId"
							className="text-xs font-medium text-zinc-600 dark:text-zinc-400"
						>
							Client ID
						</label>
						<input
							id="clientId"
							type="text"
							required
							disabled={isConfigured}
							value={isConfigured ? '' : clientId}
							onChange={(event) => setClientId(event.target.value)}
							className={inputClass}
						/>
					</div>
					<div className="flex flex-col gap-1.5">
						<label
							htmlFor="clientSecret"
							className="text-xs font-medium text-zinc-600 dark:text-zinc-400"
						>
							Client Secret
						</label>
						<input
							id="clientSecret"
							type="password"
							required
							autoComplete="off"
							disabled={isConfigured}
							placeholder={isConfigured ? '••••••••' : undefined}
							value={isConfigured ? '' : clientSecret}
							onChange={(event) => setClientSecret(event.target.value)}
							className={inputClass}
						/>
					</div>

					{provider ? (
						<button
							type="button"
							onClick={() => handleDelete(provider.providerId)}
							disabled={deletingId === provider.providerId}
							className="cursor-pointer self-start rounded-md border border-red-300 px-3 py-2 text-sm font-medium text-red-600 transition-colors hover:bg-red-50 disabled:opacity-50 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-950"
						>
							{deletingId === provider.providerId ? 'Deleting…' : 'Delete provider'}
						</button>
					) : (
						<button
							type="submit"
							disabled={submitting}
							className="cursor-pointer self-start rounded-md bg-[#76b900] px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#6aa600] disabled:opacity-50"
						>
							{submitting ? 'Saving…' : 'Save provider'}
						</button>
					)}
				</form>
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant="error"
				onClose={() => setError(null)}
			/>
			<Toast
				open={message !== null}
				message={message ?? ''}
				variant="success"
				onClose={() => setMessage(null)}
			/>
		</div>
	);
};
