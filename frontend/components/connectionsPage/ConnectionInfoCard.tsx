// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';
import { PopoverMenu } from '@/common/PopoverMenu';
import type { Connection } from '@/types/connection';

export type ConnectionInfoCardProps = {
	connection: Connection;
	onDelete?: (databaseName: string) => void;
	disabled?: boolean;
};

export const ConnectionInfoCard = ({
	connection,
	onDelete,
	disabled = false,
}: ConnectionInfoCardProps) => {
	const menuDisabled = disabled || !onDelete;

	return (
		<article className="flex h-fit flex-col rounded-lg border border-zinc-200/90 bg-white shadow-sm dark:border-zinc-700/90 dark:bg-zinc-950">
			<header className="flex w-full items-center justify-between gap-2 px-4 py-5">
				<div className="flex min-w-0 items-center gap-1">
					<Icon name={IconName.Database} className="h-5 w-5 shrink-0 text-[#76b900]" />
					<h3 className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-100">
						{connection.database_name}
					</h3>
				</div>
				<div className="relative shrink-0">
					{menuDisabled ? (
						<Icon
							name={IconName.DotsVertical}
							className="h-4 w-4 text-zinc-300 dark:text-zinc-600"
						/>
					) : (
						<PopoverMenu
							className="relative"
							items={[
								{
									label: 'Remove',
									icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
									onClick: () => onDelete(connection.database_name),
									danger: true,
								},
							]}
							trigger={({ toggle }) => (
								<button
									type="button"
									onClick={toggle}
									className="cursor-pointer rounded-md p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-800 dark:hover:text-zinc-300"
									aria-label={`Actions for ${connection.database_name}`}
								>
									<Icon name={IconName.DotsVertical} className="h-4 w-4" />
								</button>
							)}
						/>
					)}
				</div>
			</header>
		</article>
	);
};
