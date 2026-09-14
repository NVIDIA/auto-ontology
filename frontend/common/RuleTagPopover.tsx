// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { tagsApi } from '@/api/tags';
import { Button } from '@/common/Button';
import { Icon, IconName } from '@/common/icons';
import { Label } from '@/common/Label';
import { Popover } from '@/common/Popover';
import { SearchInput } from '@/common/SearchInput';
import { SkeletonBlock } from '@/common/Skeleton';
import { MAX_TAG_NAME_LENGTH } from '@/constants/tags';
import { ButtonTheme, Size } from '@/enums/button';
import { SkeletonVariant } from '@/enums/skeleton';
import type { RuleTagDraft } from '@/types/rules';
import type { TagChip } from '@/types/tags';

export type RuleTagPopoverProps = {
	/** How many items the rule would tag. */
	itemsCount: number;
	/**
	 * How many the search matched in all, when that is more than the rule can
	 * reach. Omitted — or equal to `itemsCount` — when the two are the same.
	 *
	 * They come apart for two reasons, and this panel is where a person agrees
	 * to both: the list a rule replays is capped at `GLOBAL_SEARCH_LIST_LIMIT`
	 * while the count behind the tabs is not, and a Database, a Schema or an
	 * analysis cannot carry a tag however well it matched.
	 */
	matchedCount?: number;
	/**
	 * Saves the finished rule, and resolves to what went wrong or to null when
	 * it went through.
	 *
	 * The panel holds only the two halves a person fills in — the search a rule
	 * replays belongs to the screen the panel was opened from, so that screen
	 * completes the payload and owns the write. Required rather than optional so
	 * that a caller cannot mount a form whose Apply button silently does
	 * nothing.
	 */
	onSubmit: (rule: RuleTagDraft) => Promise<string | null>;
	/** Called when the panel navigates away, so its opener can close too. */
	onNavigate?: () => void;
	/** Told whether the panel is showing, so its opener can hold still behind it. */
	onOpenChange?: (open: boolean) => void;
};

const byName = (left: TagChip, right: TagChip): number =>
	left.name.toLowerCase().localeCompare(right.name.toLowerCase());

/** Placeholder chips, at tag-name widths so the box looks like what replaces it. */
const SKELETON_TAG_WIDTHS = ['w-20', 'w-16', 'w-24', 'w-14', 'w-20'];

type RuleTagFormProps = Pick<
	RuleTagPopoverProps,
	'itemsCount' | 'matchedCount' | 'onSubmit' | 'onNavigate'
> & {
	/** Dismisses the panel this form fills. */
	onDone: () => void;
};

/**
 * What the rule will label, and what the search matched when that is more.
 *
 * Naming only the second would promise a rule that tags everything found, which
 * it cannot; naming only the first leaves a reader wondering where the rest of
 * their results went. "N of M" rather than "top N of M" because the shortfall
 * is not only the cap — an untaggable kind is dropped wherever it ranked, and
 * "top" would read as a promise that the missing rows were the worst matches.
 */
const itemsToTagLabel = (count: number, matched: number | undefined): string => {
	const noun = count === 1 ? 'item' : 'items';
	return matched != null && matched > count
		? `${count} of ${matched} ${noun} to tag`
		: `${count} ${noun} to tag`;
};

/**
 * The rule being built.
 *
 * Split from the panel so that it is mounted only while the panel is open,
 * which is what discards a half-filled rule on dismissal — there is no reset to
 * write, and the tags are read once per opening.
 */
const RuleTagForm = ({
	itemsCount,
	matchedCount,
	onSubmit,
	onNavigate,
	onDone,
}: RuleTagFormProps) => {
	const [options, setOptions] = useState<TagChip[]>([]);
	const [selected, setSelected] = useState<TagChip[]>([]);
	const [search, setSearch] = useState('');
	const [name, setName] = useState('');
	const [loading, setLoading] = useState(true);
	const [creating, setCreating] = useState(false);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		void tagsApi.getAll().then((response) => {
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load tags.');
				setOptions([]);
			} else {
				// Reduced to what a chip is, rather than held whole: the rest of a
				// tag — its dates, its author — is what this list happens to arrive
				// with, not anything the rule has a use for, and the create posts
				// these on.
				setOptions(
					(response.data ?? [])
						.map((tag) => ({ id: tag.id, name: tag.name }))
						.sort(byName),
				);
			}
			setLoading(false);
		});
		return () => {
			cancelled = true;
		};
	}, []);

	const trimmedSearch = search.trim();
	const selectedIds = new Set(selected.map((tag) => tag.id));
	const unselected = options.filter((tag) => !selectedIds.has(tag.id));
	const matches = unselected.filter((tag) =>
		tag.name.toLowerCase().includes(trimmedSearch.toLowerCase()),
	);
	// Offered only when nothing existing matches, so the row cannot compete with
	// picking a tag that is already there.
	const offerCreate =
		!loading &&
		trimmedSearch !== '' &&
		trimmedSearch.length <= MAX_TAG_NAME_LENGTH &&
		!options.some((tag) => tag.name.toLowerCase() === trimmedSearch.toLowerCase());
	// A rule with no tags applies nothing, so both halves are required.
	const canSubmit = name.trim() !== '' && selected.length > 0 && !saving;

	const handleCreateTag = async () => {
		if (!offerCreate || creating) return;
		setCreating(true);
		const response = await tagsApi.create({ name: trimmedSearch });
		setCreating(false);

		if (response.error) {
			setError(response.message ?? 'Failed to create tag.');
			return;
		}
		const created = response.data;
		if (created == null) return;

		setError(null);
		setOptions((prev) => [...prev, created].sort(byName));
		setSelected((prev) => [...prev, { id: created.id, name: created.name }].sort(byName));
		setSearch('');
	};

	/**
	 * The panel closes only once the rule is saved.
	 *
	 * A failed save leaves the form standing with what was typed in it, because
	 * that is the only copy of it — the form is unmounted on dismissal, so
	 * closing first and reporting the failure elsewhere would ask someone to
	 * fill the whole rule in again.
	 */
	const handleApply = async () => {
		if (!canSubmit) return;
		setSaving(true);
		const message = await onSubmit({ name: name.trim(), tags: selected });

		if (message !== null) {
			setError(message);
			setSaving(false);
			return;
		}
		onDone();
	};

	return (
		<>
			<div className="flex items-baseline gap-2 border-b border-zinc-200 px-4 py-3 dark:border-zinc-700">
				<Icon
					name={IconName.Lightning}
					className="h-4 w-4 shrink-0 self-center text-[#76b900]"
				/>
				<h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
					Create a New Rule Based Tag
				</h3>
				<span className="text-xs text-zinc-500 dark:text-zinc-400">
					({itemsToTagLabel(itemsCount, matchedCount)})
				</span>
			</div>

			<div className="space-y-4 px-4 py-3">
				<div>
					<label
						htmlFor="rule-name"
						className="mb-1.5 block text-sm font-semibold text-zinc-900 dark:text-zinc-100"
					>
						Rule Name
					</label>
					<input
						id="rule-name"
						type="text"
						value={name}
						onChange={(event) => setName(event.target.value)}
						placeholder="Name this rule and save it!"
						autoFocus
						// Same frame and height as the search field below it, so the two
						// controls in this panel read as one set.
						className="h-9 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500"
					/>
					<p className="mt-1.5 text-xs text-zinc-500 dark:text-zinc-400">
						Every item this search matches is labelled with these tags, and so is every
						item that matches it later.
					</p>
				</div>

				<div>
					<p className="mb-1.5 flex items-center gap-2 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						<Icon
							name={IconName.Tag}
							className="h-4 w-4 shrink-0 text-zinc-500 dark:text-zinc-400"
						/>
						Add Tag/s…
					</p>

					{selected.length > 0 ? (
						<ul className="mb-2 flex max-h-[84px] flex-wrap gap-2 overflow-y-auto [scrollbar-gutter:stable]">
							{selected.map((tag) => (
								<li key={tag.id}>
									<Label
										label={tag.name}
										onRemove={() =>
											setSelected((prev) =>
												prev.filter((item) => item.id !== tag.id),
											)
										}
									/>
								</li>
							))}
						</ul>
					) : null}

					<div className="space-y-3">
						<SearchInput
							value={search}
							onChange={setSearch}
							placeholder="Select or search to add a tag…"
							aria-label="Search tags"
							className="h-9 w-full"
						/>
						{/* Fixed rather than fitted: the panel hangs off its trigger, so a
						    box that resizes when the tags arrive — or when a search
						    narrows the list — would make the whole panel jump. Long tag
						    lists scroll in here, with the scrollbar's gutter always
						    reserved so the chips do not shift when it appears. */}
						<div className="h-[120px] overflow-y-auto rounded-lg border border-zinc-200 p-2 [scrollbar-gutter:stable] dark:border-zinc-700">
							{loading ? (
								<div className="flex flex-wrap gap-2" role="status">
									{SKELETON_TAG_WIDTHS.map((width, index) => (
										<SkeletonBlock
											key={index}
											variant={SkeletonVariant.CIRCLE}
											className={`h-[26px] ${width}`}
										/>
									))}
									<span className="sr-only">Loading tags</span>
								</div>
							) : matches.length > 0 ? (
								<ul className="flex flex-wrap gap-2">
									{matches.map((tag) => (
										<li key={tag.id}>
											<Label
												label={tag.name}
												onClick={() =>
													setSelected((prev) =>
														[...prev, tag].sort(byName),
													)
												}
											/>
										</li>
									))}
								</ul>
							) : offerCreate ? (
								<button
									type="button"
									onClick={() => {
										void handleCreateTag();
									}}
									disabled={creating}
									className="flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-[#4d7a00] transition-colors hover:bg-[#76b900]/10 disabled:cursor-default disabled:opacity-60 dark:text-[#a3d63a] dark:hover:bg-[#76b900]/15"
								>
									<Icon name={IconName.Plus} className="h-4 w-4 shrink-0" />
									{`${trimmedSearch} (Create Tag - Max. ${MAX_TAG_NAME_LENGTH})`}
								</button>
							) : (
								<p className="px-2 text-xs text-zinc-500 dark:text-zinc-400">
									{options.length === 0
										? 'No tags exist yet'
										: unselected.length === 0
											? 'Every tag is already selected'
											: 'No tag matches this search'}
								</p>
							)}
						</div>
					</div>
				</div>

				{error !== null ? (
					<div className="rounded-lg border border-red-200/90 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
						{error}
					</div>
				) : null}
			</div>

			<div className="flex items-center justify-between gap-3 border-t border-zinc-200 px-4 py-3 dark:border-zinc-700">
				<Link
					href="/settings/rules"
					onClick={onNavigate}
					className="inline-flex items-center gap-1.5 text-xs text-zinc-500 no-underline transition-colors hover:text-[#4d7a00] dark:text-zinc-400 dark:hover:text-[#a3d63a]"
				>
					<Icon name={IconName.Tag} className="h-3.5 w-3.5 shrink-0" />
					Manage Rule Based Tags
					<Icon name={IconName.Link} className="h-3.5 w-3.5 shrink-0" />
				</Link>
				<Button
					theme={ButtonTheme.Primary}
					size={Size.REGULAR}
					type="button"
					disabled={!canSubmit}
					loading={saving}
					onClick={() => {
						void handleApply();
					}}
				>
					{saving ? 'Applying…' : 'Apply'}
				</Button>
			</div>
		</>
	);
};

/**
 * The "Create a Ruled Tag" button and the panel it opens under itself.
 *
 * A panel rather than a dialog because the search it builds a rule from stays
 * on screen behind it — the results are the rule's subject, and covering them
 * with a modal would hide what is about to be tagged.
 */
export const RuleTagPopover = ({
	itemsCount,
	matchedCount,
	onSubmit,
	onNavigate,
	onOpenChange,
}: RuleTagPopoverProps) => (
	<Popover
		onOpenChange={onOpenChange}
		panelClassName="w-[480px] max-w-[calc(100vw-2rem)] max-h-[calc(100vh-8rem)] overflow-y-auto"
		trigger={({ toggle }) => (
			<Button
				theme={ButtonTheme.Secondary}
				size={Size.REGULAR}
				type="button"
				onClick={toggle}
				iconPosition="left"
			>
				<Icon name={IconName.Lightning} className="h-4 w-4" />
				Create a Ruled Tag
			</Button>
		)}
	>
		{({ close }) => (
			<RuleTagForm
				itemsCount={itemsCount}
				matchedCount={matchedCount}
				onSubmit={onSubmit}
				onNavigate={onNavigate}
				onDone={close}
			/>
		)}
	</Popover>
);
