// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useState } from 'react';
import { Placeholders } from '@/assets/images/placeholders';
import { Button } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { Size, ButtonTheme } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { ToastVariant } from '@/enums/toast';
import { ConnectionsInfoCardView } from '@/components/connectionsPage/ConnectionsInfoCardView';
import { NewConnectionsModal } from '@/components/connectionsPage/NewConnectionsModal';
import { ConfirmModal } from '@/common/modal';
import { Toast } from '@/common/Toast';
import { Icon, IconName } from '@/common/icons';
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
	const [ssoFederationPending, setSsoFederationPending] = useState<string | null>(null);
	const [ssoError, setSsoError] = useState<string | null>(null);

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

	const handleSsoFederationChange = async (databaseName: string, enabled: boolean) => {
		setSsoError(null);
		setSsoFederationPending(databaseName);

		// Optimistic update — apply immediately so the checkbox doesn't snap back.
		setConnections((prev) =>
			prev.map((c) =>
				c.database_name === databaseName
					? { ...c, connection: { ...c.connection, sso_federation: enabled } }
					: c,
			),
		);

		const res = await connectionsApi.setSsoFederation(databaseName, enabled);
		setSsoFederationPending(null);

		if (res.error) {
			// Revert optimistic update on failure.
			setConnections((prev) =>
				prev.map((c) =>
					c.database_name === databaseName
						? { ...c, connection: { ...c.connection, sso_federation: !enabled } }
						: c,
				),
			);
			setSsoError(res.message ?? 'Failed to update connection.');
		}
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
			<div className="flex h-full w-full flex-1 flex-col gap-4 p-6">
				<div className="h-8 w-48 animate-pulse rounded-md bg-zinc-100 dark:bg-zinc-800/70" />
				<ConnectionsInfoCardView connections={[]} loading />
			</div>
		);
	}

	if (error) {
		return (
			<div className="flex h-full w-full flex-1 flex-col items-center justify-center gap-3 px-6">
				<p className="text-sm text-red-600 dark:text-red-400">{error}</p>
				<Button
					theme={ButtonTheme.Secondary}
					size={Size.REGULAR}
					type="button"
					onClick={() => {
						setLoading(true);
						void fetchConnections().finally(() => setLoading(false));
					}}
				>
					Retry
				</Button>
			</div>
		);
	}

	return (
		<div className="flex h-full w-full min-w-0 flex-1 flex-col items-start">
			{connections.length === 0 ? (
				<EmptyState
					variant={EmptyStateVariant.Borderless}
					illustration={<Placeholders.NoConnections />}
					title="No Connections Created Yet"
					action={{
						label: 'Create New Connection',
						icon: IconName.Database,
						onClick: handleCreateConnection,
					}}
				/>
			) : (
				<div className="flex w-full flex-col items-start gap-5">
					<div className="flex w-full justify-end">
						<Button
							theme={ButtonTheme.Primary}
							size={Size.REGULAR}
							type="button"
							onClick={handleCreateConnection}
							iconPosition="left"
							shadow
						>
							<Icon name={IconName.Database} className="h-4 w-4" />
							Create New Connection
						</Button>
					</div>
					<ConnectionsInfoCardView
						connections={connections}
						onDelete={handleDeleteRequest}
						onSsoFederationChange={(databaseName, enabled) => {
							void handleSsoFederationChange(databaseName, enabled);
						}}
						ssoFederationPending={ssoFederationPending}
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

			<Toast
				open={ssoError !== null}
				message={ssoError ?? ''}
				variant={ToastVariant.Error}
				onClose={() => setSsoError(null)}
			/>
		</div>
	);
};
