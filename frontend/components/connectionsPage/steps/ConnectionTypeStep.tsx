// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';
import { ConnectionType, connectionDisplayName } from '@/enums/connection';

export type ConnectionTypeStepProps = {
	onSelect: (type: ConnectionType) => void;
};

const CONNECTOR_TYPES = Object.values(ConnectionType).sort((a, b) =>
	connectionDisplayName[a].localeCompare(connectionDisplayName[b]),
);

export const ConnectionTypeStep = ({ onSelect }: ConnectionTypeStepProps) => (
	<div className="flex flex-col gap-3 p-2">
		<div className="grid grid-cols-2 gap-3">
			{CONNECTOR_TYPES.map((type) => (
				<button
					key={type}
					type="button"
					data-testid={`connection-type-${type}`}
					onClick={() => onSelect(type)}
					className="flex h-[150px] cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-zinc-200 bg-white px-4 py-6 shadow-sm transition-colors hover:border-[#76b900]/50 hover:bg-[#76b900]/5 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-[#76b900]/40"
				>
					<Icon name={IconName.Database} className="h-8 w-8 text-[#76b900]" />
					<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
						{connectionDisplayName[type]}
					</span>
				</button>
			))}
		</div>
	</div>
);
