// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Suspense, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Button } from '@/common/Button';
import { Icon, IconName } from '@/common/icons';
import { authClient } from '@/auth/auth-client';
import {
	deliverLoopbackCallback,
	followOAuthRedirect,
	isLoopbackCallback,
} from '@/auth/oauth-loopback';
import { OauthStatus } from '@/app/oauth/OauthStatus';
import { ButtonTheme, Size } from '@/enums/button';

const ConsentForm = () => {
	const params = useSearchParams();
	const [submitting, setSubmitting] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [outcome, setOutcome] = useState<'granted' | 'denied' | null>(null);
	const clientId = params.get('client_id') ?? 'the MCP client';
	const scopes = (params.get('scope') ?? '')
		.split(' ')
		.map((scope) => scope.trim())
		.filter(Boolean);

	const submit = async (accept: boolean) => {
		setSubmitting(true);
		setError(null);
		const result = await authClient.oauth2.consent({ accept });
		if (result.data?.redirect && result.data.url) {
			let callback: URL | null = null;
			try {
				callback = new URL(result.data.url, window.location.origin);
			} catch {
				callback = null;
			}
			if (callback && isLoopbackCallback(callback)) {
				await deliverLoopbackCallback(callback.toString());
				setOutcome(accept ? 'granted' : 'denied');
				setSubmitting(false);
				return;
			}
			followOAuthRedirect(result.data.url);
			return;
		}
		setError(result.error?.message ?? 'Could not complete authorization.');
		setSubmitting(false);
	};

	if (outcome === 'granted') {
		return (
			<OauthStatus title="Thank you">
				Authentication successful. You can close this page.
			</OauthStatus>
		);
	}

	if (outcome === 'denied') {
		return (
			<OauthStatus title="Authorization cancelled">
				You can close this page and return to the application that started sign-in.
			</OauthStatus>
		);
	}

	return (
		<div className="flex w-full max-w-md flex-col gap-5">
			<div className="space-y-2 text-center">
				<h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100">
					Authorize MCP access
				</h1>
				<p className="text-sm text-zinc-600 dark:text-zinc-400">
					{clientId} is requesting access to Auto Ontology as your account.
				</p>
			</div>

			{scopes.length ? (
				<div className="rounded-md border border-zinc-200 p-4 dark:border-zinc-700">
					<p className="mb-2 text-xs font-medium text-zinc-600 dark:text-zinc-400">
						Requested access
					</p>
					<ul className="space-y-1 text-sm text-zinc-800 dark:text-zinc-200">
						{scopes.map((scope) => (
							<li key={scope}>{scope}</li>
						))}
					</ul>
				</div>
			) : null}

			{error ? <p className="text-center text-xs text-red-500">{error}</p> : null}

			<div className="flex justify-center gap-3">
				<Button
					theme={ButtonTheme.Minimal}
					size={Size.REGULAR}
					type="button"
					disabled={submitting}
					onClick={() => submit(false)}
				>
					Cancel
				</Button>
				<Button
					theme={ButtonTheme.Primary}
					size={Size.REGULAR}
					type="button"
					disabled={submitting}
					onClick={() => submit(true)}
				>
					{submitting ? 'Authorizing…' : 'Authorize'}
				</Button>
			</div>
		</div>
	);
};

const ConsentPage = () => (
	<div className="flex h-full w-full items-center justify-center bg-white p-6 dark:bg-zinc-950">
		<div className="flex flex-col items-center gap-6">
			<div className="flex items-center gap-2">
				<Icon name={IconName.NvidiaLogo} className="h-6 w-6" />
				<span className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
					Auto Ontology
				</span>
			</div>
			<Suspense fallback={null}>
				<ConsentForm />
			</Suspense>
		</div>
	</div>
);

export default ConsentPage;
