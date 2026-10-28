// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';

import { Button, SelectButton } from '@/common/Button';
import {
	SEARCH_TYPE_ICON,
	SEARCH_TYPE_TAB_LABEL,
	SEARCH_TYPE_TAB_ORDER,
} from '@/common/globalSearchMeta';
import { Icon, IconName } from '@/common/icons';
import { Popover } from '@/common/Popover';
import { SearchInput } from '@/common/SearchInput';
import { Toggle } from '@/common/Toggle';
import { DEFAULT_DISCOVERY_FILTERS } from '@/constants/discovery';
import { UNTAGGED_TAG_FILTER } from '@/constants/search';
import { ButtonTheme, SelectButtonTheme, Size } from '@/enums/button';
import { PopoverAlign } from '@/enums/popover';
import type { SearchObjectType } from '@/enums/search';
import { tagQueries } from '@/lib/queries/tags';
import type { DiscoveryFilters } from '@/types/discovery';
import { DataFilter } from './DataFilter';

export type DiscoveryFiltersPanelProps = {
	/** The filters being edited, which are not yet the ones a search ran with. */
	filters: DiscoveryFilters;
	onChange: (filters: DiscoveryFilters) => void;
	onReset: () => void;
};

/**
 * How many filters stand at something other than their default.
 *
 * Counted rather than read off a separate "changed" copy of the filters: the
 * panel offers a reset, and what it resets to is
 * `DEFAULT_DISCOVERY_FILTERS` — so that object is the only definition of
 * untouched there needs to be.
 */
export const changedDiscoveryFilterCount = (filters: DiscoveryFilters): number =>
	(filters.objects === null ? 0 : 1) +
	(filters.tags === null ? 0 : 1) +
	(filters.data === null ? 0 : 1) +
	(filters.description === DEFAULT_DISCOVERY_FILTERS.description ? 0 : 1);

/**
 * Whether these filters exclude everything, so there is nothing to search for.
 *
 * Reachable only through Deselect All, and a real answer rather than an error:
 * a search restricted to no kind of object, to no tag, or to no part of the
 * catalog has nowhere to look. `DiscoveryView` uses it to answer such a
 * search itself, since the request has no way to ask for nothing.
 */
export const discoveryFiltersMatchNothing = (filters: DiscoveryFilters): boolean =>
	filters.objects?.length === 0 || filters.tags?.length === 0 || filters.data?.length === 0;

/**
 * The bulk button's two states, shared by both pickers so they cannot drift.
 *
 * It is one button rather than two because the two are never both useful:
 * with everything ticked there is nothing left to select, and with nothing
 * ticked nothing left to clear.
 */
const bulkActionLabel = (allSelected: boolean): string =>
	allSelected ? 'Deselect All' : 'Select All';

/** What the closed select field says about a selection. */
const selectionSummary = (selected: string[] | null): string => {
	if (selected === null) return 'All';
	if (selected.length === 0) return 'None';
	return `Selected (${selected.length})`;
};

/**
 * The width every filter's control is laid out in.
 *
 * Fixed rather than fitted so the controls line up down the panel instead of
 * each one ending wherever its label left off. It also bounds the select
 * fields, whose summary grows with the selection — "Selected (10)" is wider
 * than "All", and a field that resized as boxes were ticked would shunt its
 * own label around.
 */
const CONTROL_WIDTH = 'w-32';

type FilterRowProps = {
	icon: IconName;
	title: string;
	changed: boolean;
	children: ReactNode;
};

/**
 * One filter: what it is on the left, the control for it on the right.
 *
 * Every row the same way round, whether its control is a select field or a
 * toggle, so the panel reads as one list of settings rather than as a form
 * whose fields happen to sit near some switches.
 */
const FilterRow = ({ icon, title, changed, children }: FilterRowProps) => (
	<div className="flex items-center justify-between gap-2 px-4 py-1">
		<span className="flex min-w-0 items-center gap-1.5">
			<Icon name={icon} className="h-4 w-4 shrink-0 text-secondary dark:text-zinc-400" />
			<span className="truncate text-sm text-body dark:text-zinc-300">{title}</span>
			{changed && (
				<span
					className="h-1.5 w-1.5 shrink-0 self-start rounded-full bg-red-500"
					aria-hidden
				/>
			)}
		</span>
		<span className={`flex shrink-0 justify-end ${CONTROL_WIDTH}`}>{children}</span>
	</div>
);

/**
 * Which kinds of object a search may hit.
 *
 * Null is every kind, so the list opens with every box ticked — an unfiltered
 * panel showing eleven cleared boxes would read as "nothing will be searched".
 * Ticking the last missing box collapses back to null rather than storing all
 * eleven, so "untouched" has one representation and the reset button has one
 * thing to compare against.
 *
 * Clearing the last ticked box is allowed and means what it says: no kind is
 * searched, so nothing is found. That is what makes Deselect All honest.
 */
const ObjectsFilter = ({
	selected,
	onChange,
}: {
	selected: SearchObjectType[] | null;
	onChange: (selected: SearchObjectType[] | null) => void;
}) => {
	const effective = selected ?? SEARCH_TYPE_TAB_ORDER;
	const allSelected = selected === null;

	const toggleType = (type: SearchObjectType) => {
		const next = effective.includes(type)
			? effective.filter((item) => item !== type)
			: [...effective, type];

		onChange(SEARCH_TYPE_TAB_ORDER.every((item) => next.includes(item)) ? null : next);
	};

	return (
		<Popover
			className="w-full"
			// Right-aligned and wider than the field it hangs off, rather than
			// stretched to it: the field is sized to the panel's column and a
			// list of ten object names is not.
			align={PopoverAlign.Right}
			panelClassName="w-56 max-h-80 overflow-y-auto p-1"
			trigger={({ open, toggle }) => (
				<SelectButton
					theme={SelectButtonTheme.SelectField}
					onClick={toggle}
					aria-expanded={open}
					aria-label="Filter by object type"
				>
					<span className="truncate">{selectionSummary(selected)}</span>
					<Icon
						name={IconName.ChevronRight}
						className="h-4 w-4 shrink-0 rotate-90 text-secondary dark:text-zinc-400"
					/>
				</SelectButton>
			)}
		>
			{() => (
				<div className="flex flex-col">
					<div className="flex items-center justify-between gap-2 px-2 py-1">
						<span className="text-xs font-medium text-secondary dark:text-zinc-400">
							Filter By
						</span>
						<Button
							theme={ButtonTheme.Minimal}
							size={Size.SMALL}
							type="button"
							onClick={() => onChange(allSelected ? [] : null)}
						>
							{bulkActionLabel(allSelected)}
						</Button>
					</div>
					{SEARCH_TYPE_TAB_ORDER.map((type) => {
						const checked = effective.includes(type);
						return (
							<label
								key={type}
								className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm text-body hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
							>
								<input
									type="checkbox"
									checked={checked}
									onChange={() => toggleType(type)}
									className="h-4 w-4 shrink-0 accent-[#76b900]"
								/>
								<Icon
									name={SEARCH_TYPE_ICON[type]}
									className="h-4 w-4 shrink-0 text-secondary dark:text-zinc-400"
								/>
								<span className="truncate">{SEARCH_TYPE_TAB_LABEL[type]}</span>
							</label>
						);
					})}
				</div>
			)}
		</Popover>
	);
};

type TagOptionProps = {
	label: string;
	checked: boolean;
	onToggle: () => void;
};

const TagOption = ({ label, checked, onToggle }: TagOptionProps) => (
	<label className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm text-body hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800">
		<input
			type="checkbox"
			checked={checked}
			onChange={onToggle}
			className="h-4 w-4 shrink-0 accent-[#76b900]"
		/>
		<span className="truncate">{label}</span>
	</label>
);

/**
 * Which tags a hit must carry, with "no tags at all" as one of the options.
 *
 * Reads like the object filter: null is every option and opens with every box
 * ticked, and ticking the last missing box collapses back to null. The
 * collapse is exact here only because `(Blanks)` is one of the options —
 * "carrying one of these tags, or carrying none" over the whole vocabulary is
 * every object there is, which is what
 * `test_tags_and_the_untagged_sentinel_together_narrow_nothing` pins down in
 * the DAL. Tick every real tag *without* it and the search genuinely does
 * narrow, so that state is a selection and is sent as one.
 *
 * `(Blanks)` sits first because it is the option somebody comes here for: the
 * question this filter answers most often is which objects nobody has labelled
 * yet. It is sent as `UNTAGGED_TAG_FILTER`, which the backend reads; a tag
 * cannot collide with it, since the others are sent as UUIDs.
 */
const TagsFilter = ({
	selected,
	onChange,
}: {
	selected: string[] | null;
	onChange: (selected: string[] | null) => void;
}) => {
	const [search, setSearch] = useState('');
	// The vocabulary the rest of the app already holds — the layout prefetches
	// it, so this panel usually opens on the tags rather than on "Loading".
	const { data: options = [], isPending, error } = useQuery(tagQueries.vocabulary());

	// Nothing is offered until the vocabulary is in hand. Mid-read the list is
	// empty, so `(Blanks)` would stand alone as the only option and ticking it
	// would read as having ticked everything — storing null, the opposite of
	// the narrowing the reader asked for.
	const ready = !isPending && error === null;
	const query = search.trim().toLowerCase();
	const matches = options.filter((tag) => tag.name.toLowerCase().includes(query));
	// Searchable like the tags are, so the option can be reached by typing
	// rather than only by scrolling past a long vocabulary to the top.
	const untaggedMatches = ready && (query === '' || '(blanks)'.includes(query));
	// A failed read and a pending one each leave the list empty for a reason
	// worth naming, and neither is "no tag matches this search".
	const emptyMessage =
		error !== null
			? 'Could not load tags'
			: isPending
				? 'Loading tags…'
				: options.length === 0
					? 'No tags exist yet'
					: matches.length === 0 && !untaggedMatches
						? 'No tag matches this search'
						: null;

	// Every option there is, in the order they are listed. Built from the
	// vocabulary as it stands, so a tag created elsewhere while this is open
	// widens "all" rather than leaving a stale list behind — and a selection
	// already collapsed to null stays correct, since null is not a list.
	const everyOption = [UNTAGGED_TAG_FILTER, ...options.map((tag) => tag.id)];
	const effective = selected ?? everyOption;
	const allSelected = selected === null;

	const toggle = (value: string) => {
		const held = effective.filter((item) => everyOption.includes(item));
		const next = held.includes(value)
			? held.filter((item) => item !== value)
			: [...held, value];
		onChange(everyOption.every((option) => next.includes(option)) ? null : next);
	};

	// Named by count rather than by name: tag names run to 25 characters and
	// the field is one column of a narrow panel, so a single name would be
	// truncated to nothing recognisable and two would not fit at all.
	const summary = selectionSummary(selected);

	return (
		<Popover
			className="w-full"
			align={PopoverAlign.Right}
			panelClassName="w-56 p-1"
			trigger={({ open, toggle: togglePanel }) => (
				<SelectButton
					theme={SelectButtonTheme.SelectField}
					onClick={togglePanel}
					aria-expanded={open}
					aria-label="Filter by tag"
				>
					<span className="truncate">{summary}</span>
					<Icon
						name={IconName.ChevronRight}
						className="h-4 w-4 shrink-0 rotate-90 text-secondary dark:text-zinc-400"
					/>
				</SelectButton>
			)}
		>
			{() => (
				<div className="flex flex-col">
					<div className="flex items-center justify-between gap-2 px-2 py-1">
						<span className="text-xs font-medium text-secondary dark:text-zinc-400">
							Filter By
						</span>
						<Button
							theme={ButtonTheme.Minimal}
							size={Size.SMALL}
							type="button"
							onClick={() => onChange(allSelected ? [] : null)}
							disabled={!ready}
						>
							{bulkActionLabel(allSelected)}
						</Button>
					</div>
					<div className="px-1 pb-1">
						<SearchInput
							value={search}
							onChange={setSearch}
							placeholder="Search tags…"
							aria-label="Search tags"
							className="h-8 w-full"
						/>
					</div>
					<div className="max-h-56 overflow-y-auto [scrollbar-gutter:stable]">
						{untaggedMatches && (
							<TagOption
								label="(Blanks)"
								checked={effective.includes(UNTAGGED_TAG_FILTER)}
								onToggle={() => toggle(UNTAGGED_TAG_FILTER)}
							/>
						)}
						{ready &&
							matches.map((tag) => (
								<TagOption
									key={tag.id}
									label={tag.name}
									checked={effective.includes(tag.id)}
									onToggle={() => toggle(tag.id)}
								/>
							))}
						{emptyMessage !== null && (
							<p className="px-2 py-1.5 text-xs text-secondary dark:text-zinc-400">
								{emptyMessage}
							</p>
						)}
					</div>
				</div>
			)}
		</Popover>
	);
};

/**
 * The filters Discovery offers.
 *
 * Owner, usage, documentation, zones, status and the count filters the
 * equivalent screen elsewhere shows are deliberately absent: nothing in this
 * deployment's search request carries them, so each would be a control that
 * changes what the panel reports and not what the query matches.
 *
 * Synonym matching is the other way round — the request carries it, but it is
 * left on rather than offered, so it is not a filter this panel holds.
 */
export const DiscoveryFiltersPanel = ({
	filters,
	onChange,
	onReset,
}: DiscoveryFiltersPanelProps) => {
	const changedCount = changedDiscoveryFilterCount(filters);

	return (
		<div className="flex h-full min-h-0 flex-col">
			<div className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
				<span className="flex items-center gap-1.5">
					<span className="text-base font-semibold text-heading dark:text-zinc-100">
						Filters
					</span>
					<Icon name={IconName.Filter} className="h-4 w-4 text-body dark:text-zinc-300" />
				</span>
				{changedCount > 0 && (
					<Button
						theme={ButtonTheme.Minimal}
						size={Size.SMALL}
						type="button"
						onClick={onReset}
					>
						Reset All ({changedCount})
					</Button>
				)}
			</div>
			<div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto py-3">
				<FilterRow
					icon={IconName.Exploration}
					title="Objects"
					changed={filters.objects !== null}
				>
					<ObjectsFilter
						selected={filters.objects}
						onChange={(objects) => onChange({ ...filters, objects })}
					/>
				</FilterRow>
				<FilterRow icon={IconName.Tag} title="Tags" changed={filters.tags !== null}>
					<TagsFilter
						selected={filters.tags}
						onChange={(tags) => onChange({ ...filters, tags })}
					/>
				</FilterRow>
				<FilterRow icon={IconName.Table} title="Data" changed={filters.data !== null}>
					<DataFilter
						selected={filters.data}
						onChange={(data) => onChange({ ...filters, data })}
					/>
				</FilterRow>
				<FilterRow
					icon={IconName.Pencil}
					title="Search In Description"
					changed={filters.description !== DEFAULT_DISCOVERY_FILTERS.description}
				>
					<Toggle
						checked={filters.description}
						onChange={(description) => onChange({ ...filters, description })}
						aria-label="Search in description"
					/>
				</FilterRow>
			</div>
		</div>
	);
};
