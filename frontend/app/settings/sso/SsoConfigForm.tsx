// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { authApi, type SsoProvider } from '@/api/auth';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { ToastVariant } from '@/enums/toast';
import { Toast } from '@/common/Toast';

// Only one provider is supported; its id is a fixed constant. It's the DB key
// and the providerId segment of the native callback (/api/auth/sso/callback/sso).
const SSO_PROVIDER_ID = 'sso';

const inputClass =
	'w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm text-body outline-none transition-colors placeholder:text-secondary focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-disabled dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:disabled:bg-zinc-800/50 dark:disabled:text-zinc-400';

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
				<h1 className="mb-1 text-lg font-semibold text-heading dark:text-zinc-100">
					Single Sign-On (OIDC)
				</h1>
				<p className="mb-4 text-xs text-secondary">
					Configure a single OpenID Connect provider. Endpoints are discovered from the
					issuer. The client secret is stored securely and never shown again.
				</p>

				<form onSubmit={handleSubmit} className="flex flex-col gap-4">
					<div className="flex flex-col gap-1.5">
						<label
							htmlFor="issuer"
							className="text-xs font-medium text-body dark:text-zinc-400"
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
							className="text-xs font-medium text-body dark:text-zinc-400"
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
							className="text-xs font-medium text-body dark:text-zinc-400"
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
						<div className="self-start">
							<Button
								theme={ButtonTheme.DangerSubtle}
								size={Size.REGULAR}
								type="button"
								onClick={() => handleDelete(provider.provider_id)}
								disabled={deletingId === provider.provider_id}
							>
								{deletingId === provider.provider_id
									? 'Deleting…'
									: 'Delete provider'}
							</Button>
						</div>
					) : (
						<div className="self-start">
							<Button
								theme={ButtonTheme.Primary}
								size={Size.REGULAR}
								type="submit"
								disabled={submitting}
							>
								{submitting ? 'Saving…' : 'Save provider'}
							</Button>
						</div>
					)}
				</form>
			</div>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setError(null)}
			/>
			<Toast
				open={message !== null}
				message={message ?? ''}
				variant={ToastVariant.Success}
				onClose={() => setMessage(null)}
			/>
		</div>
	);
};
