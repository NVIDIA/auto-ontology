// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiTokensApi, type ApiToken } from '@/api/apiTokens';
import type { ResponseWithError } from '@/api/types';
import { Button, CopyButton } from '@/common/Button';
import { formatDate } from '@/common/date';
import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { ConfirmModal, Modal } from '@/common/modal';
import { Table } from '@/common/Table';
import { Toast } from '@/common/Toast';
import { ButtonTheme, Size } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { ToastVariant } from '@/enums/toast';
import type { TableColumn } from '@/types/table';

// `undefined` days = never expires, which is what an unattended script wants.
const EXPIRY_OPTIONS: { label: string; days?: number }[] = [
	{ label: 'Never' },
	{ label: '30 days', days: 30 },
	{ label: '90 days', days: 90 },
	{ label: '1 year', days: 365 },
];

const inputClassName =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 outline-none focus:border-[#76b900] focus:ring-1 focus:ring-[#76b900] dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100';

const isExpired = (token: ApiToken): boolean =>
	token.expires_at != null && new Date(token.expires_at).getTime() <= Date.now();

/**
 * Self-service API tokens for scripting. The plaintext token exists only in the
 * creation response, so it is surfaced once in a dismissible panel and never
 * refetched — the list can only ever show the leading characters.
 */
export const ApiTokensView = () => {
	const [tokens, setTokens] = useState<ApiToken[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const [creating, setCreating] = useState(false);
	const [createOpen, setCreateOpen] = useState(false);
	const [name, setName] = useState('');
	const [expiryIndex, setExpiryIndex] = useState(0);
	const [createError, setCreateError] = useState<string | null>(null);
	const [freshToken, setFreshToken] = useState<string | null>(null);

	const [revoking, setRevoking] = useState<ApiToken | null>(null);
	const [revokeBusy, setRevokeBusy] = useState(false);
	const [revokeError, setRevokeError] = useState<string | null>(null);

	const applyList = useCallback((res: ResponseWithError<ApiToken[]>) => {
		if (res.error) {
			setError(res.message ?? 'Failed to load API tokens.');
			return;
		}
		setError(null);
		setTokens(res);
	}, []);

	const refresh = useCallback(async () => applyList(await apiTokensApi.list()), [applyList]);

	useEffect(() => {
		let cancelled = false;
		void apiTokensApi.list().then((res) => {
			if (cancelled) return;
			applyList(res);
			setLoading(false);
		});
		return () => {
			cancelled = true;
		};
	}, [applyList]);

	const openCreate = () => {
		setName('');
		setExpiryIndex(0);
		setCreateError(null);
		setCreateOpen(true);
	};

	const handleCreate = async () => {
		const trimmed = name.trim();
		if (!trimmed) {
			setCreateError('Give the token a name so you can recognise it later.');
			return;
		}

		setCreating(true);
		setCreateError(null);
		const res = await apiTokensApi.create(trimmed, EXPIRY_OPTIONS[expiryIndex].days);
		setCreating(false);

		if (res.error) {
			setCreateError(res.message ?? 'Failed to create the token.');
			return;
		}

		setCreateOpen(false);
		setFreshToken(res.token);
		await refresh();
	};

	const handleRevoke = async () => {
		if (!revoking) return;
		setRevokeBusy(true);
		setRevokeError(null);
		const res = await apiTokensApi.revoke(revoking.id);
		setRevokeBusy(false);

		if (res.error) {
			setRevokeError(res.message ?? 'Failed to revoke the token.');
			return;
		}

		setRevoking(null);
		await refresh();
	};

	const columns: TableColumn<ApiToken>[] = [
		{
			key: 'name',
			header: 'Name',
			truncate: true,
			cell: (token) => (
				<span className="font-medium text-zinc-900 dark:text-zinc-100">
					{token.name ?? 'Unnamed'}
				</span>
			),
		},
		{
			key: 'start',
			header: 'Token',
			width: 'w-44',
			nowrap: true,
			cell: (token) => (
				<code className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
					{token.start ? `${token.start}…` : '—'}
				</code>
			),
		},
		{
			key: 'created_at',
			header: 'Created',
			width: 'w-32',
			nowrap: true,
			cell: (token) => formatDate(token.created_at),
		},
		{
			key: 'last_request',
			header: 'Last used',
			width: 'w-32',
			nowrap: true,
			cell: (token) => (token.last_request ? formatDate(token.last_request) : 'Never'),
		},
		{
			key: 'expires_at',
			header: 'Expires',
			width: 'w-32',
			nowrap: true,
			cell: (token) => {
				if (!token.expires_at) return 'Never';
				return (
					<span
						className={isExpired(token) ? 'text-red-600 dark:text-red-400' : undefined}
					>
						{formatDate(token.expires_at)}
					</span>
				);
			},
		},
		{
			key: 'actions',
			header: <span className="sr-only">Actions</span>,
			width: 'w-16',
			nowrap: true,
			cell: (token) => (
				<Button
					theme={ButtonTheme.IconDanger}
					size={Size.SMALL}
					iconOnly
					aria-label={`Revoke ${token.name ?? 'token'}`}
					onClick={() => {
						setRevokeError(null);
						setRevoking(token);
					}}
				>
					<Icon name={IconName.Trash} className="h-4 w-4" />
				</Button>
			),
		},
	];

	return (
		<main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]">
			<div className="w-full max-w-4xl space-y-5">
				<div className="flex items-start justify-between gap-4">
					<div className="min-w-0">
						<h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
							API Tokens
						</h1>
						<p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
							Call the Auto Ontology API from scripts without signing in. A token acts
							as you — it can do exactly what your account can, and nothing more.
						</p>
					</div>
					<Button theme={ButtonTheme.Primary} size={Size.SMALL} onClick={openCreate}>
						<Icon name={IconName.Plus} className="mr-1.5 h-4 w-4" />
						New token
					</Button>
				</div>

				{freshToken ? (
					<div className="rounded-lg border border-[#76b900]/40 bg-[#76b900]/5 p-4">
						<div className="flex items-start justify-between gap-3">
							<div className="min-w-0">
								<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
									Copy your token now
								</h2>
								<p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
									Auto Ontology stores only a hash of it, so this is the one and
									only time it can be shown. Send it as the <code>x-api-key</code>{' '}
									header.
								</p>
							</div>
							<Button
								theme={ButtonTheme.Minimal}
								size={Size.SMALL}
								onClick={() => setFreshToken(null)}
							>
								Done
							</Button>
						</div>
						<div className="group mt-3 flex items-center gap-2 rounded-md bg-zinc-900 px-3 py-2 dark:bg-zinc-950">
							<code className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-100">
								{freshToken}
							</code>
							<CopyButton text={freshToken} className="opacity-100" />
						</div>
					</div>
				) : null}

				{loading ? (
					<p className="text-sm text-zinc-500 dark:text-zinc-400">Loading…</p>
				) : null}

				{!loading && tokens.length === 0 ? (
					<EmptyState
						icon={IconName.Key}
						title="No API tokens yet"
						description="Create one to authenticate scripts and scheduled jobs against the Auto Ontology API."
						variant={EmptyStateVariant.Dashed}
					/>
				) : null}

				{!loading && tokens.length > 0 ? (
					<Table columns={columns} rows={tokens} rowKey={(token) => token.id} />
				) : null}
			</div>

			<Modal
				open={createOpen}
				onClose={() => setCreateOpen(false)}
				className="w-full max-w-md"
			>
				<div className="space-y-4 p-5">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						New API token
					</h2>

					<label className="block space-y-1.5">
						<span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
							Name
						</span>
						<input
							className={inputClassName}
							value={name}
							maxLength={64}
							placeholder="nightly-report script"
							onChange={(event) => setName(event.target.value)}
						/>
					</label>

					<label className="block space-y-1.5">
						<span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
							Expires
						</span>
						<select
							className={inputClassName}
							value={expiryIndex}
							onChange={(event) => setExpiryIndex(Number(event.target.value))}
						>
							{EXPIRY_OPTIONS.map((option, index) => (
								<option key={option.label} value={index}>
									{option.label}
								</option>
							))}
						</select>
					</label>

					{createError ? (
						<p className="text-xs text-red-600 dark:text-red-400">{createError}</p>
					) : null}

					<div className="flex justify-end gap-2">
						<Button
							theme={ButtonTheme.Secondary}
							size={Size.SMALL}
							onClick={() => setCreateOpen(false)}
						>
							Cancel
						</Button>
						<Button
							theme={ButtonTheme.Primary}
							size={Size.SMALL}
							disabled={creating}
							onClick={() => {
								void handleCreate();
							}}
						>
							{creating ? 'Creating…' : 'Create token'}
						</Button>
					</div>
				</div>
			</Modal>

			<ConfirmModal
				open={revoking !== null}
				title="Revoke token"
				message={
					<>
						Any script using <strong>{revoking?.name ?? 'this token'}</strong> will stop
						working immediately. This cannot be undone.
					</>
				}
				confirmLabel="Revoke"
				confirming={revokeBusy}
				error={revokeError}
				onConfirm={handleRevoke}
				onCancel={() => setRevoking(null)}
			/>

			<Toast
				open={error !== null}
				message={error ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setError(null)}
			/>
		</main>
	);
};
