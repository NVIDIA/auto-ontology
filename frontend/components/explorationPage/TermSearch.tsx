// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState, type KeyboardEvent } from 'react';

import { termsApi } from '@/api/terms';
import { CertificationBadge } from '@/common/CertificationBadge';
import { Icon, IconName } from '@/common/icons';
import { SearchInput } from '@/common/SearchInput';
import { Text } from '@/common/Text';
import { TermSearchMatch } from '@/enums/search';
import { TextVariant } from '@/enums/text';
import { useDebouncedValue } from '@/hooks/useDebouncedValue';
import type { TermSearchResult } from '@/types/terms';

/** Long enough to skip most keystrokes, short enough to feel live. */
const SEARCH_DEBOUNCE_MS = 250;
const RESULT_LIMIT = 10;

type SearchResponse = {
	query: string;
	results: TermSearchResult[];
	semantic: boolean;
	error: string | null;
};

type TermSearchProps = {
	/** Called with the picked term's id; the caller selects it in the graph. */
	onPick: (termId: string) => void;
};

/**
 * The Exploration search box on the Semantic layer.
 *
 * Asks `GET /terms/search`, which returns terms whose name contains the text
 * first and then terms the semantic index ranks close to it, so "client" can
 * surface *Customer*. Results are not limited to the nodes the graph has
 * loaded: a pick outside them still opens in the side panel.
 *
 * Arrow keys move through the dropdown, Enter picks, Escape closes it.
 */
export const TermSearch = ({ onPick }: TermSearchProps) => {
	const [query, setQuery] = useState('');
	const debounced = useDebouncedValue(query.trim(), SEARCH_DEBOUNCE_MS);
	// The last response, tagged with the text it answers. Results show only
	// while that text is still what's in the box, so a slow response for an
	// older query can never replace a newer one on screen.
	const [response, setResponse] = useState<SearchResponse | null>(null);
	const [open, setOpen] = useState(false);
	const [highlighted, setHighlighted] = useState(0);
	const rootRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (debounced === '') return;
		let cancelled = false;
		void termsApi.search(debounced, RESULT_LIMIT).then((res) => {
			if (cancelled) return;
			setResponse(
				res.error || res.data == null
					? {
							query: debounced,
							results: [],
							semantic: true,
							error: res.message ?? 'Search failed',
						}
					: {
							query: debounced,
							results: res.data,
							semantic: res.semantic,
							error: null,
						},
			);
			setHighlighted(0);
		});
		return () => {
			cancelled = true;
		};
	}, [debounced]);

	const current = debounced !== '' && response?.query === debounced ? response : null;
	const results = current?.results ?? [];
	const semantic = current?.semantic ?? true;
	const error = current?.error ?? null;
	const loading = debounced !== '' && current == null;

	// Close the dropdown on a click anywhere else.
	useEffect(() => {
		if (!open) return;
		const onPointerDown = (event: PointerEvent) => {
			if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
		};
		document.addEventListener('pointerdown', onPointerDown);
		return () => document.removeEventListener('pointerdown', onPointerDown);
	}, [open]);

	const pick = (term: TermSearchResult) => {
		setQuery('');
		setOpen(false);
		onPick(term.id);
	};

	const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
		if (event.key === 'Escape') {
			setOpen(false);
			return;
		}
		if (results.length === 0) return;
		if (event.key === 'ArrowDown') {
			event.preventDefault();
			setOpen(true);
			setHighlighted((index) => (index + 1) % results.length);
		} else if (event.key === 'ArrowUp') {
			event.preventDefault();
			setHighlighted((index) => (index - 1 + results.length) % results.length);
		} else if (event.key === 'Enter') {
			event.preventDefault();
			const term = results[highlighted];
			if (term) pick(term);
		}
	};

	const showDropdown = open && query.trim() !== '';
	const firstMeaning = results.findIndex((term) => term.match === TermSearchMatch.Meaning);

	return (
		<div ref={rootRef} className="relative w-96" onKeyDown={handleKeyDown}>
			<SearchInput
				value={query}
				onChange={(value) => {
					setQuery(value);
					setOpen(true);
				}}
				placeholder="Search terms by name or meaning…"
				aria-label="Search terms in exploration"
				className="h-10 bg-white shadow-md dark:bg-zinc-900"
			/>
			{showDropdown && (
				<div className="absolute top-12 w-full overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
					{loading && results.length === 0 ? (
						<p className="px-3 py-2.5 text-sm text-secondary dark:text-zinc-400">
							Searching…
						</p>
					) : error != null ? (
						<p className="px-3 py-2.5 text-sm text-red-700 dark:text-red-300">
							{error}
						</p>
					) : results.length === 0 && debounced !== '' ? (
						<p className="px-3 py-2.5 text-sm text-secondary dark:text-zinc-400">
							No terms match.
						</p>
					) : (
						<ul
							role="listbox"
							aria-label="Matching terms"
							className="max-h-[calc(100dvh-10rem)] overflow-y-auto py-1"
						>
							{results.map((term, index) => (
								<li key={term.id} role="presentation">
									{index === firstMeaning && (
										<p className="border-t border-zinc-100 px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wide text-secondary first:border-t-0 dark:border-zinc-800 dark:text-zinc-400">
											Related by meaning
										</p>
									)}
									<button
										type="button"
										role="option"
										aria-selected={index === highlighted}
										onMouseEnter={() => setHighlighted(index)}
										onClick={() => pick(term)}
										className={`flex w-full items-start gap-2.5 px-3 py-2 text-left ${
											index === highlighted
												? 'bg-zinc-100 dark:bg-zinc-800'
												: 'bg-transparent'
										}`}
									>
										<Icon
											name={IconName.Terms}
											className="mt-0.5 h-4 w-4 shrink-0 text-[#76b900]"
										/>
										<span className="min-w-0 flex-1">
											<span className="flex items-center gap-2">
												<span className="min-w-0 font-medium">
													<Text
														text={term.name}
														variant={TextVariant.Body}
													/>
												</span>
												<CertificationBadge
													status={term.certification}
													iconOnly
													decorative
													focusable={false}
												/>
											</span>
											{term.description ? (
												<span className="mt-0.5 line-clamp-2 block text-xs text-secondary dark:text-zinc-400">
													{term.description}
												</span>
											) : null}
										</span>
										{term.similarity != null && (
											<span
												className="shrink-0 pt-0.5 text-[11px] tabular-nums text-secondary dark:text-zinc-400"
												title="Similarity from the semantic index"
											>
												{Math.round(term.similarity * 100)}%
											</span>
										)}
									</button>
								</li>
							))}
						</ul>
					)}
					{!semantic && results.length > 0 && (
						<p className="border-t border-zinc-100 px-3 py-2 text-xs text-secondary dark:border-zinc-800 dark:text-zinc-400">
							Matching by meaning is unavailable right now; showing name matches only.
						</p>
					)}
				</div>
			)}
		</div>
	);
};
