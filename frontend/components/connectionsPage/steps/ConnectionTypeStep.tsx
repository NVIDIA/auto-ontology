// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { OptionCard } from '@/components/connectionsPage/steps/OptionCard';
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
				<OptionCard key={type} onClick={() => onSelect(type)}>
					<Icon
						name={IconName.Database}
						className="h-8 w-8 text-body dark:text-zinc-300"
					/>
					<span className="text-sm font-medium text-heading dark:text-zinc-100">
						{connectionDisplayName[type]}
					</span>
				</OptionCard>
			))}
		</div>
	</div>
);
