// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useState } from 'react';
import { Placeholders } from '@/assets/images/placeholders';
import { ConnectionsInfoCardView } from '@/components/connectionsPage/ConnectionsInfoCardView';
import { NewConnectionsModal } from '@/components/connectionsPage/NewConnectionsModal';
import { ConfirmModal } from '@/components/ConfirmModal';
import { Icon, IconName } from '@/components/icons';
import { connectionsApi } from '@/api/connections';
import type { Connection } from '@/types/connection';

export const ConnectionsView = () => {
	const [connections, setConnections] = useState<Connection[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [connectionModalOpen, setConnectionModalOpen] = useState(false);
	const [deletingConnection, setDeletingConnection] = useState<string | null>(null);
	const [deleting, setDeleting] = useState(false);
	const [deleteError, setDeleteError] = useState<string | null>(null);

	const fetchConnections = useCallback(async () => {
		try {
			setError(null);
			const res = await connectionsApi.getAll();
			if (res.error) {
				setError(res.message ?? 'Failed to load connections.');
				setConnections([]);
				return;
			}
			setConnections(res.data ?? []);
		} catch {
			setError('Failed to load connections.');
			setConnections([]);
		}
	}, []);

	useEffect(() => {
		void (async () => {
			setLoading(true);
			await fetchConnections();
			setLoading(false);
		})();
	}, [fetchConnections]);

	const handleCreateConnection = () => {
		setConnectionModalOpen(true);
	};

	const handleConnectionModalClose = () => {
		setConnectionModalOpen(false);
	};

	const handleConnectionModalConfirm = () => {
		handleConnectionModalClose();
		void fetchConnections();
	};

	const handleDeleteRequest = (databaseName: string) => {
		setDeleteError(null);
		setDeletingConnection(databaseName);
	};

	const handleDeleteClose = () => {
		if (deleting) return;
		setDeletingConnection(null);
		setDeleteError(null);
	};

	const handleDeleteConfirm = async () => {
		if (deletingConnection == null) return;
		setDeleting(true);
		setDeleteError(null);
		const res = await connectionsApi.delete(deletingConnection);
		setDeleting(false);

		if (res.error) {
			setDeleteError(res.message ?? 'Failed to delete connection.');
			return;
		}

		setConnections((prev) => prev.filter((c) => c.database_name !== deletingConnection));
		setDeletingConnection(null);
	};

	if (loading) {
		return (
			<div className="flex h-full min-h-[240px] w-full flex-1 items-center justify-center">
				<div
					className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
					role="status"
					aria-label="Loading connections"
				/>
			</div>
		);
	}

	if (error) {
		return (
			<div className="flex h-full w-full flex-1 flex-col items-center justify-center gap-3 px-6">
				<p className="text-sm text-red-600 dark:text-red-400">{error}</p>
				<button
					type="button"
					onClick={() => {
						setLoading(true);
						void fetchConnections().finally(() => setLoading(false));
					}}
					className="cursor-pointer rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800"
				>
					Retry
				</button>
			</div>
		);
	}

	return (
		<div
			data-testid="connections-page"
			className="flex h-full w-full min-w-0 flex-1 flex-col items-start"
		>
			{connections.length === 0 ? (
				<div className="flex w-full flex-1 flex-col items-center justify-center py-16">
					<Placeholders.NoConnections />
					<h2 className="mt-6 text-base font-semibold text-zinc-900 dark:text-zinc-100">
						No Connections Created Yet
					</h2>
					<button
						type="button"
						onClick={handleCreateConnection}
						className="mt-6 flex cursor-pointer items-center gap-2 rounded-lg bg-[#76b900] px-4 py-2.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#6aa500]"
					>
						<Icon name={IconName.Database} className="h-4 w-4" />
						Create New Connection
					</button>
				</div>
			) : (
				<div className="flex w-full flex-col items-start gap-5">
					<div className="flex w-full justify-end">
						<button
							type="button"
							onClick={handleCreateConnection}
							className="flex cursor-pointer items-center justify-center gap-2 rounded-lg bg-[#76b900] px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#6aa500]"
						>
							<Icon name={IconName.Database} className="h-4 w-4" />
							Create New Connection
						</button>
					</div>
					<ConnectionsInfoCardView
						connections={connections}
						onDelete={handleDeleteRequest}
					/>
				</div>
			)}

			<ConfirmModal
				open={deletingConnection !== null}
				title="Remove connection"
				message={
					deletingConnection ? (
						<>
							Are you sure you want to remove <strong>{deletingConnection}</strong>?
							Its entire subgraph will be deleted. This action cannot be undone.
						</>
					) : null
				}
				confirmLabel="Remove"
				onConfirm={handleDeleteConfirm}
				onCancel={handleDeleteClose}
				confirming={deleting}
				error={deleteError}
			/>

			<NewConnectionsModal
				key={String(connectionModalOpen)}
				open={connectionModalOpen}
				onConfirm={handleConnectionModalConfirm}
				onCancel={handleConnectionModalClose}
			/>
		</div>
	);
};
