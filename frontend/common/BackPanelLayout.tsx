// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useState, type ReactNode } from 'react';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';

export type BackPanelLayoutProps = {
	panel: ReactNode;
	children: ReactNode;
	panelAriaLabel?: string;
	expandAriaLabel?: string;
	collapseAriaLabel?: string;
	className?: string;
	defaultCollapsed?: boolean;
	expandedWidthClass?: string;
	collapsedWidthClass?: string;
};

const PanelCollapseToggle = ({
	collapsed,
	onToggle,
	expandAriaLabel,
	collapseAriaLabel,
}: {
	collapsed: boolean;
	onToggle: () => void;
	expandAriaLabel: string;
	collapseAriaLabel: string;
}) => (
	<Button
		theme={ButtonTheme.IconNeutral}
		size={Size.SMALL}
		iconOnly
		type="button"
		onClick={onToggle}
		aria-label={collapsed ? expandAriaLabel : collapseAriaLabel}
	>
		<svg
			width="16"
			height="16"
			viewBox="0 0 16 16"
			fill="none"
			xmlns="http://www.w3.org/2000/svg"
		>
			{collapsed ? (
				<>
					<path
						d="M6 3L11 8L6 13"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
					<line
						x1="13.25"
						y1="3"
						x2="13.25"
						y2="13"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
					/>
				</>
			) : (
				<>
					<path
						d="M10 3L5 8L10 13"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
						strokeLinejoin="round"
					/>
					<line
						x1="2.75"
						y1="3"
						x2="2.75"
						y2="13"
						stroke="currentColor"
						strokeWidth="1.5"
						strokeLinecap="round"
					/>
				</>
			)}
		</svg>
	</Button>
);

export const BackPanelLayout = ({
	panel,
	children,
	panelAriaLabel,
	expandAriaLabel = 'Expand panel',
	collapseAriaLabel = 'Collapse panel',
	className = '',
	defaultCollapsed = false,
	expandedWidthClass = 'w-[296px]',
	collapsedWidthClass = 'w-8',
}: BackPanelLayoutProps) => {
	const [collapsed, setCollapsed] = useState(defaultCollapsed);
	const toggleCollapsed = useCallback(() => setCollapsed((c) => !c), []);

	return (
		<div className={`flex h-full w-full bg-white dark:bg-zinc-950 ${className}`}>
			<aside
				className={`flex h-full shrink-0 flex-col border-r border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950 ${collapsed ? collapsedWidthClass : expandedWidthClass}`}
			>
				{collapsed ? (
					<div className="flex h-full w-8 min-w-[32px] flex-col items-center bg-transparent pt-1">
						<PanelCollapseToggle
							collapsed
							onToggle={toggleCollapsed}
							expandAriaLabel={expandAriaLabel}
							collapseAriaLabel={collapseAriaLabel}
						/>
					</div>
				) : (
					<div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-transparent">
						<div className="flex h-6 shrink-0 items-center justify-end border-b border-zinc-200/80 px-1 dark:border-zinc-700">
							<PanelCollapseToggle
								collapsed={false}
								onToggle={toggleCollapsed}
								expandAriaLabel={expandAriaLabel}
								collapseAriaLabel={collapseAriaLabel}
							/>
						</div>
						<div
							className="flex min-h-0 flex-1 flex-col overflow-hidden"
							aria-label={panelAriaLabel}
						>
							{panel}
						</div>
					</div>
				)}
			</aside>
			<div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">{children}</div>
		</div>
	);
};
