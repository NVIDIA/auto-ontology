// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CSSProperties } from 'react';

import { Text } from '@/common/Text';

// A label carries a name, so it is capped and clipped rather than allowed to
// grow: one oversized value (a description that ended up in a name field, say)
// would otherwise take over the whole chip list.
const LABEL_BASE =
	'inline-flex max-w-[min(24rem,100%)] items-center rounded-full border px-3 py-1 text-xs font-medium';
const LABEL_GREEN =
	'border-[#76b900]/40 bg-[#76b900]/10 text-[#4d7a00] dark:border-[#76b900]/30 dark:bg-[#76b900]/15 dark:text-[#a3d63a]';
const LABEL_GREEN_HOVER =
	'hover:border-[#76b900]/70 hover:bg-[#76b900]/20 hover:text-[#3d6200] dark:hover:border-[#76b900]/60 dark:hover:bg-[#76b900]/25 dark:hover:text-[#b6e05a]';
const LABEL_NEUTRAL =
	'border-zinc-200 bg-zinc-100 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300';
const LABEL_MUTED =
	'border-zinc-200 bg-zinc-100 text-zinc-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-500';

function zoneLabelStyle(color: string): CSSProperties {
	return { backgroundColor: `${color}26`, color, borderColor: `${color}60` };
}

export type LabelProps = {
	label: string;
	/** When provided the label becomes an interactive button; otherwise it is read-only. */
	onClick?: () => void;
	/** Accent color (hex) for zone labels; falls back to the brand-green palette when omitted. */
	color?: string | null;
	/** Muted grey style used for disabled zones. */
	muted?: boolean;
};

export const Label = ({ label, onClick, color, muted = false }: LabelProps) => {
	const usesZoneColor = !muted && color != null && color !== '';
	const interactive = onClick != null;
	const palette = muted ? LABEL_MUTED : usesZoneColor ? LABEL_NEUTRAL : LABEL_GREEN;
	const className = [
		LABEL_BASE,
		palette,
		interactive ? 'cursor-pointer transition-colors' : '',
		interactive && !usesZoneColor && !muted ? LABEL_GREEN_HOVER : '',
	]
		.filter(Boolean)
		.join(' ');
	const style = usesZoneColor ? zoneLabelStyle(color as string) : undefined;
	// Clipping is what makes the cap above win over a long value, and a clipped
	// chip reveals itself in the same tooltip the tags elsewhere use.
	const content = <Text text={label} />;

	if (interactive) {
		return (
			<button type="button" onClick={onClick} className={className} style={style}>
				{content}
			</button>
		);
	}
	return (
		<span className={className} style={style}>
			{content}
		</span>
	);
};
