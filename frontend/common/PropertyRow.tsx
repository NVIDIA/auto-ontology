// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { CopyButton } from '@/common/Button';

export type PropertyRowProps = {
	label: string;
	value: string;
	monospace?: boolean;
	className?: string;
};

export const PropertyRow = ({ label, value, monospace = false, className }: PropertyRowProps) => (
	<div
		className={`group relative overflow-hidden rounded-lg bg-zinc-900 dark:bg-zinc-950 ${className ?? ''}`}
	>
		<div className="flex items-center justify-between border-b border-zinc-700 px-3 py-1.5">
			<span className="text-xs font-medium text-zinc-400">{label}</span>
			{value !== '' && <CopyButton text={value} />}
		</div>
		<p
			className={`p-3 text-xs leading-relaxed wrap-anywhere text-zinc-100 ${
				monospace ? 'font-mono' : ''
			}`}
		>
			{value || '—'}
		</p>
	</div>
);
