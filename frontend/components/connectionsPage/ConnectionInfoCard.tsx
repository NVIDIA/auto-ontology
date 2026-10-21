// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import { Icon, IconName } from '@/common/icons';
import { PopoverMenu } from '@/common/PopoverMenu';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { ConnectionType } from '@/enums/connection';
import type { Connection, ConnectionUserToggle } from '@/types/connection';

export type ConnectionInfoCardProps = {
	connection: Connection;
	onEdit?: (connection: Connection) => void;
	onDelete?: (databaseName: string) => void;
	onUserToggleChange?: (
		databaseName: string,
		toggle: ConnectionUserToggle,
		enabled: boolean,
	) => void;
	/** Disables the toggle while its request is in flight. */
	userTogglePending?: boolean;
	disabled?: boolean;
};

export const ConnectionInfoCard = ({
	connection,
	onEdit,
	onDelete,
	onUserToggleChange,
	userTogglePending = false,
	disabled = false,
}: ConnectionInfoCardProps) => {
	const menuItems = [
		...(onEdit == null
			? []
			: [
					{
						label: 'Edit',
						icon: <Icon name={IconName.Pencil} className="h-3.5 w-3.5" />,
						onClick: () => onEdit(connection),
					},
				]),
		...(onDelete == null
			? []
			: [
					{
						label: 'Remove',
						icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
						onClick: () => onDelete(connection.database_name),
						danger: true,
					},
				]),
	];
	const menuDisabled = disabled || menuItems.length === 0;
	// Only Databricks (SSO federation) and ClickHouse (EXECUTE AS) can run chat
	// queries as the signed-in user.
	const params = connection.connection;
	const userToggle =
		params.type === ConnectionType.DATABRICKS
			? {
					key: 'sso_federation' as const,
					testId: 'sso-federation',
					on: params.sso_federation === true,
					label: 'Authenticate as Signed-In User',
					onText: 'Chat runs SQL with each user’s own Databricks privileges.',
					offText: 'Chat runs SQL with this connection’s stored access token.',
				}
			: params.type === ConnectionType.CLICKHOUSE
				? {
						key: 'execute_as_user' as const,
						testId: 'execute-as-user',
						on: params.execute_as_user === true,
						label: 'Run Chat Queries as Signed-In User',
						onText: 'Chat queries run as the signed-in user’s ClickHouse account (email local part). The stored user must hold IMPERSONATE.',
						offText: 'Chat queries run as the stored user.',
					}
				: null;
	const showUserToggle = userToggle != null && onUserToggleChange != null;

	return (
		<article className="flex h-fit flex-col rounded-lg border border-zinc-200/90 bg-white shadow-sm dark:border-zinc-700/90 dark:bg-zinc-950">
			<header className="flex w-full items-center justify-between gap-2 px-4 py-5">
				<div className="flex min-w-0 items-center gap-1">
					<Icon
						name={IconName.Database}
						className="h-5 w-5 shrink-0 text-body dark:text-zinc-300"
					/>
					<Text
						as="h3"
						text={connection.database_name}
						variant={TextVariant.Subheading}
					/>
				</div>
				<div className="relative shrink-0">
					{menuDisabled ? (
						<Icon
							name={IconName.DotsVertical}
							className="h-4 w-4 text-disabled dark:text-zinc-600"
						/>
					) : (
						<PopoverMenu
							className="relative"
							items={menuItems}
							trigger={({ toggle }) => (
								<Button
									theme={ButtonTheme.IconNeutral}
									size={Size.SMALL}
									iconOnly
									type="button"
									onClick={toggle}
									aria-label={`Actions for ${connection.database_name}`}
								>
									<Icon name={IconName.DotsVertical} className="h-4 w-4" />
								</Button>
							)}
						/>
					)}
				</div>
			</header>
			{showUserToggle && (
				<div className="border-t border-zinc-200/90 px-4 py-3 dark:border-zinc-700/90">
					<label className="flex cursor-pointer items-start gap-2">
						<input
							type="checkbox"
							checked={userToggle.on}
							disabled={disabled || userTogglePending}
							onChange={({ target }) =>
								onUserToggleChange(
									connection.database_name,
									userToggle.key,
									target.checked,
								)
							}
							data-testid={`${userToggle.testId}-${connection.database_name}`}
							className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer rounded border-zinc-300 accent-[#76b900] disabled:cursor-default dark:border-zinc-600"
						/>
						<span className="min-w-0">
							<span className="block text-xs font-medium text-heading dark:text-zinc-200">
								{userToggle.label}
							</span>
							<span className="block text-xs text-secondary dark:text-zinc-400">
								{userToggle.on ? userToggle.onText : userToggle.offText}
							</span>
						</span>
					</label>
				</div>
			)}
		</article>
	);
};
