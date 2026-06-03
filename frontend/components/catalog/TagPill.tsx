// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tag } from '@kui/foundations-react';
import type { OmTagLabel } from '@/types/openmetadata';

type TagColor = 'blue' | 'green' | 'red' | 'yellow' | 'purple' | 'teal' | 'gray';

const colorFor = (fqn: string): TagColor => {
	if (fqn === 'PII.Sensitive') return 'red';
	if (fqn.startsWith('PII.')) return 'yellow';
	if (fqn.startsWith('Tier.')) return 'purple';
	if (fqn.startsWith('Certification.')) return 'green';
	return 'blue';
};

const shortLabel = (fqn: string): string => {
	const parts = fqn.split('.');
	return parts.length >= 2 ? `${parts[0]}: ${parts.slice(1).join('.')}` : fqn;
};

export const TagPill = ({ label, dense = false }: { label: OmTagLabel; dense?: boolean }) => {
	const suggested = label.state === 'Suggested';

	return (
		<Tag
			readOnly
			color={colorFor(label.tagFQN)}
			kind={suggested ? 'outline' : 'solid'}
			density={dense ? 'compact' : undefined}
			title={`${label.tagFQN}  ·  ${label.source}/${label.labelType}  ·  ${label.state}`}
		>
			{shortLabel(label.tagFQN)}
			{suggested ? (
				<span className="ml-1 text-[9px] uppercase tracking-wide opacity-70">
					suggested
				</span>
			) : null}
		</Tag>
	);
};

export const TagsRow = ({ tags }: { tags: OmTagLabel[] | undefined | null }) => {
	if (!tags || tags.length === 0) {
		return <span className="text-xs text-[var(--text-color-subtle)]">—</span>;
	}
	return (
		<div className="flex flex-wrap gap-1.5">
			{tags.map((t) => (
				<TagPill key={`${t.tagFQN}-${t.source}-${t.state}`} label={t} />
			))}
		</div>
	);
};
