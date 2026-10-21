// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { ConnectionInfoCard } from '@/components/connectionsPage/ConnectionInfoCard';
import { EmptyState } from '@/common/EmptyState';
import { IconName } from '@/common/icons';
import { SkeletonCard } from '@/common/Skeleton';
import { EmptyStateVariant } from '@/enums/emptyState';
import type { Connection, ConnectionUserToggle } from '@/types/connection';

type ConnectionsInfoCardViewProps = {
	connections: Connection[];
	loading?: boolean;
	onEdit?: (connection: Connection) => void;
	onDelete?: (databaseName: string) => void;
	onUserToggleChange?: (
		databaseName: string,
		toggle: ConnectionUserToggle,
		enabled: boolean,
	) => void;
	/** Database name whose run-as-signed-in-user request is currently in flight. */
	userTogglePending?: string | null;
};

export const ConnectionsInfoCardView = ({
	connections,
	loading = false,
	onEdit,
	onDelete,
	onUserToggleChange,
	userTogglePending = null,
}: ConnectionsInfoCardViewProps) => {
	if (loading) {
		return (
			<div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
				{Array.from({ length: 9 }, (_, i) => (
					<SkeletonCard key={i} className="min-h-[148px]" />
				))}
			</div>
		);
	}

	if (connections.length === 0) {
		return (
			<EmptyState
				variant={EmptyStateVariant.Borderless}
				icon={IconName.Connection}
				title="No Results"
			/>
		);
	}

	return (
		<div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
			{connections.map((connection) => (
				<ConnectionInfoCard
					key={connection.database_name}
					connection={connection}
					onEdit={onEdit}
					onDelete={onDelete}
					onUserToggleChange={onUserToggleChange}
					userTogglePending={userTogglePending === connection.database_name}
				/>
			))}
		</div>
	);
};
