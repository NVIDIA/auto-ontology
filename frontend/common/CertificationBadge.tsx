// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Icon, IconName } from '@/common/icons';
import { CertificationStatus } from '@/enums/certification';

/** Status text the badge shows, which a control wrapping one names itself with. */
export const CERTIFICATION_LABEL: Record<CertificationStatus, string> = {
	[CertificationStatus.Pending]: 'Pending Approval',
	[CertificationStatus.Partial]: 'Partially Certified',
	[CertificationStatus.Certified]: 'Certified',
};

const PILL_CLASS: Record<CertificationStatus, string> = {
	[CertificationStatus.Certified]:
		'border-[#76b900]/40 bg-[#76b900]/10 text-[#4d7a00] dark:border-[#76b900]/30 dark:bg-[#76b900]/15 dark:text-[#a3d63a]',
	[CertificationStatus.Partial]:
		'border-amber-300/60 bg-amber-100 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/15 dark:text-amber-300',
	[CertificationStatus.Pending]:
		'border-zinc-200 bg-zinc-100 text-secondary dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-400',
};

const ICON_CLASS: Record<CertificationStatus, string> = {
	[CertificationStatus.Certified]: 'text-[#76b900]',
	[CertificationStatus.Partial]: 'text-amber-500',
	[CertificationStatus.Pending]: 'text-secondary dark:text-zinc-500',
};

export type CertificationBadgeProps = {
	status: CertificationStatus;
	/** Render only the colored icon (no pill / text). */
	iconOnly?: boolean;
	/**
	 * Set when an enclosing control already names the status and owns the focus.
	 * The icon then renders as a plain graphic: naming itself would repeat the
	 * control, and taking focus would nest a tab stop inside it.
	 */
	decorative?: boolean;
	/**
	 * Clear inside an element that is already a single tab stop, such as a
	 * clickable row. The icon keeps its name — which an enclosing widget folds
	 * into its own — but stops adding a tab stop of its own to every row.
	 */
	focusable?: boolean;
	className?: string;
};

export const CertificationBadge = ({
	status,
	iconOnly = false,
	decorative = false,
	focusable = true,
	className = '',
}: CertificationBadgeProps) => {
	const title = CERTIFICATION_LABEL[status];

	if (iconOnly) {
		if (decorative) {
			return (
				<Icon
					name={IconName.Certification}
					aria-hidden
					className={`h-4 w-4 shrink-0 ${ICON_CLASS[status]} ${className}`}
				/>
			);
		}

		return (
			<span
				role="img"
				aria-label={title}
				// A status carried by colour alone has to be readable without a
				// mouse, which on its own the badge answers by taking focus to
				// reveal its tooltip.
				tabIndex={focusable ? 0 : undefined}
				className={`group relative inline-flex shrink-0 rounded outline-none focus-visible:ring-2 focus-visible:ring-[#76b900]/50 ${className}`}
			>
				<Icon
					name={IconName.Certification}
					aria-hidden
					className={`h-4 w-4 ${ICON_CLASS[status]}`}
				/>
				<span
					aria-hidden
					className="pointer-events-none absolute bottom-full left-1/2 z-[1000] mb-1.5 -translate-x-1/2 whitespace-nowrap rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs font-medium text-body opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
				>
					{title}
				</span>
			</span>
		);
	}

	return (
		<span
			title={title}
			className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-xs font-medium ${PILL_CLASS[status]} ${className}`}
		>
			<Icon name={IconName.Certification} className="h-3.5 w-3.5" />
			{title}
		</span>
	);
};
