// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OmTagLabel } from '@/types/openmetadata';

const isPii = (fqn: string): boolean => fqn.startsWith('PII.');
const isSensitive = (fqn: string): boolean => fqn === 'PII.Sensitive';

const colorFor = (label: OmTagLabel): string => {
	if (isSensitive(label.tagFQN)) {
		return 'bg-red-100 text-red-800 ring-red-300 dark:bg-red-950/50 dark:text-red-200 dark:ring-red-800';
	}
	if (isPii(label.tagFQN)) {
		return 'bg-amber-100 text-amber-900 ring-amber-300 dark:bg-amber-950/50 dark:text-amber-200 dark:ring-amber-800';
	}
	if (label.tagFQN.startsWith('Tier.')) {
		return 'bg-violet-100 text-violet-800 ring-violet-300 dark:bg-violet-950/50 dark:text-violet-200 dark:ring-violet-800';
	}
	if (label.tagFQN.startsWith('Certification.')) {
		return 'bg-emerald-100 text-emerald-800 ring-emerald-300 dark:bg-emerald-950/50 dark:text-emerald-200 dark:ring-emerald-800';
	}
	return 'bg-sky-100 text-sky-800 ring-sky-300 dark:bg-sky-950/50 dark:text-sky-200 dark:ring-sky-800';
};

const shortLabel = (fqn: string): string => {
	const parts = fqn.split('.');
	return parts.length >= 2 ? `${parts[0]}: ${parts.slice(1).join('.')}` : fqn;
};

export const TagPill = ({ label, dense = false }: { label: OmTagLabel; dense?: boolean }) => {
	const auto = label.labelType === 'Generated' || label.labelType === 'Automated';
	const suggested = label.state === 'Suggested';
	const cls = colorFor(label);

	return (
		<span
			title={`${label.tagFQN}  ·  ${label.source}/${label.labelType}  ·  ${label.state}`}
			className={`inline-flex items-center gap-1 rounded-full ring-1 ring-inset ${cls} ${
				dense ? 'px-1.5 py-0 text-[10px]' : 'px-2 py-0.5 text-xs'
			} font-medium ${suggested ? 'italic opacity-90' : ''}`}
		>
			{auto ? (
				<svg
					viewBox="0 0 20 20"
					fill="currentColor"
					className={dense ? 'h-2.5 w-2.5' : 'h-3 w-3'}
					aria-hidden
				>
					<path
						fillRule="evenodd"
						d="M9.732 1.064a1 1 0 0 1 1.736 0l1.66 2.871a1 1 0 0 0 .369.369l2.871 1.66a1 1 0 0 1 0 1.736l-2.871 1.66a1 1 0 0 0-.369.369l-1.66 2.871a1 1 0 0 1-1.736 0l-1.66-2.871a1 1 0 0 0-.369-.369l-2.871-1.66a1 1 0 0 1 0-1.736l2.871-1.66a1 1 0 0 0 .369-.369l1.66-2.871z"
						clipRule="evenodd"
					/>
				</svg>
			) : null}
			{shortLabel(label.tagFQN)}
			{suggested ? (
				<span className="text-[9px] uppercase tracking-wide opacity-70">suggested</span>
			) : null}
		</span>
	);
};

export const TagsRow = ({ tags }: { tags: OmTagLabel[] | undefined | null }) => {
	if (!tags || tags.length === 0) {
		return <span className="text-xs text-zinc-400 dark:text-zinc-600">—</span>;
	}
	return (
		<div className="flex flex-wrap gap-1.5">
			{tags.map((t) => (
				<TagPill key={`${t.tagFQN}-${t.source}-${t.state}`} label={t} />
			))}
		</div>
	);
};
