// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { ReactNode } from 'react';

import { Button } from '@/common/Button';
import { formatDate } from '@/common/date';
import { Icon, IconName } from '@/common/icons';
import { Label } from '@/common/Label';
import { Modal } from '@/common/modal';
import { SEARCH_TYPE_TAB_LABEL } from '@/common/globalSearchMeta';
import { ButtonTheme, Size } from '@/enums/button';
import type { Rule } from '@/types/rules';

import { RuleAuthor } from './RuleAuthor';

/**
 * Which kinds of object a rule labels, as the search tab it was saved from
 * named them.
 *
 * Null for a rule saved from the All tab, which narrowed to no kind — that one
 * labels everything its term matches, now and as the catalog grows, which is
 * the part of a rule most worth stating.
 */
export const scopeLabel = (rule: Rule): string => {
	const objects = rule.filters.objects;
	if (objects == null || objects.length === 0) return 'All types';
	return objects.map((type) => SEARCH_TYPE_TAB_LABEL[type] ?? type).join(', ');
};

const Row = ({ icon, label, children }: { icon: IconName; label: string; children: ReactNode }) => (
	<div className="flex items-start gap-2 text-sm">
		<Icon
			name={icon}
			className="mt-0.5 h-4 w-4 shrink-0 text-zinc-500 dark:text-zinc-400"
			aria-hidden
		/>
		<span className="font-semibold text-zinc-900 dark:text-zinc-100">{label}:</span>
		<span className="min-w-0 text-zinc-600 dark:text-zinc-300">{children}</span>
	</div>
);

/**
 * Both timestamps are serialised from one row by one serialiser, so a rule
 * nobody has renamed has them byte-identical and no date parsing is needed to
 * tell — the same test the tag list applies. Showing an edit that never
 * happened, and an editor there is none of, would be worse than showing
 * nothing.
 */
const wasRenamed = (rule: Rule): boolean => rule.modified !== rule.created;

/**
 * Everything a rule is, for a card that shows only its name and its author.
 *
 * Read-only, apart from the name, which is renamed from the card's own menu
 * rather than here: the search a rule replays is what it *is*, and re-pointing
 * it at another one would silently change which objects it labels. That is a
 * new rule, made from the search the way this one was.
 */
export const RuleInformationModal = ({
	rule,
	onClose,
}: {
	rule: Rule | null;
	onClose: () => void;
}) => (
	<Modal open={rule != null} onClose={onClose} className="w-full max-w-lg">
		<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
			<div className="flex min-w-0 items-center gap-2">
				<Icon name={IconName.Lightning} className="h-5 w-5 shrink-0 text-[#76b900]" />
				<h2 className="min-w-0 truncate text-base font-semibold text-zinc-900 dark:text-zinc-100">
					{rule?.name}
				</h2>
			</div>
			<Button
				theme={ButtonTheme.IconNeutral}
				size={Size.SMALL}
				iconOnly
				type="button"
				onClick={onClose}
				aria-label="Close rule information"
			>
				<Icon name={IconName.Close} className="h-4 w-4" />
			</Button>
		</header>

		{rule != null ? (
			<div className="divide-y divide-zinc-200 dark:divide-zinc-700">
				<div className="space-y-3 px-5 py-4 text-sm">
					<div className="flex items-center gap-2">
						<Icon
							name={IconName.Users}
							className="h-4 w-4 shrink-0 text-zinc-500 dark:text-zinc-400"
							aria-hidden
						/>
						<span className="font-semibold text-zinc-900 dark:text-zinc-100">
							Created By
						</span>
						<RuleAuthor author={rule.created_by_user} />
					</div>

					{/* Only for a rule that has been renamed, since that is the only
					    edit there is: an unrenamed rule has no editor and no second
					    date, and a row saying so would read as an edit. */}
					{wasRenamed(rule) ? (
						<div className="flex items-center gap-2">
							<Icon
								name={IconName.Pencil}
								className="h-4 w-4 shrink-0 text-zinc-500 dark:text-zinc-400"
								aria-hidden
							/>
							<span className="font-semibold text-zinc-900 dark:text-zinc-100">
								Renamed By
							</span>
							<RuleAuthor author={rule.modified_by_user} />
							<span className="ml-auto shrink-0 text-xs text-zinc-500 dark:text-zinc-400">
								{formatDate(rule.modified)}
							</span>
						</div>
					) : null}
				</div>

				<div className="px-5 py-4">
					<Row icon={IconName.Tag} label="Tags">
						<span className="flex flex-wrap gap-1.5">
							{rule.tags.map((tag) => (
								<Label key={tag.id} label={tag.name} />
							))}
						</span>
					</Row>
				</div>

				{/* The search, as much of it as is worth reading back: the term it
				    replays and the kinds of object it labels. The rest of what was
				    saved with it — the match option, and whether descriptions and
				    synonyms were searched — is how the search was run rather than
				    anything about this rule, and reading it changes no decision
				    anyone makes here. */}
				<div className="space-y-2 px-5 py-4">
					<Row icon={IconName.Exploration} label="Searched Word">
						{`'${rule.search_term}'`}
					</Row>
					<Row icon={IconName.Column} label="Applies To">
						{scopeLabel(rule)}
					</Row>
				</div>
			</div>
		) : null}
	</Modal>
);
