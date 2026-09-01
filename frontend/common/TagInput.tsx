// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import {
	forwardRef,
	useCallback,
	useImperativeHandle,
	useMemo,
	useRef,
	useState,
	type ClipboardEvent,
	type KeyboardEvent,
} from 'react';
import { Text } from '@/common/Text';

export type TagInputHandle = {
	/** Forces any pending text in the input to be committed as a tag. */
	flush: () => string[];
	focus: () => void;
};

export type TagInputProps = {
	value: string[];
	onChange: (next: string[]) => void;
	/** Placeholder shown only when the input is empty AND no tags exist. */
	placeholder?: string;
	/** Keys that commit the current input as a tag. Defaults to Enter/Comma/Semicolon/Tab. */
	commitKeys?: readonly string[];
	/** Characters that split pasted text into multiple tags. */
	splitPattern?: RegExp;
	/** Optional id (mirrored to the inner input for label association). */
	id?: string;
	/** Aria-label for the input field. */
	ariaLabel?: string;
	disabled?: boolean;
	/** Optional max number of tags. Extra ones are dropped. */
	maxTags?: number;
	/** Optional custom validator; return false to reject the tag. */
	validate?: (tag: string) => boolean;
	autoFocus?: boolean;
};

const DEFAULT_COMMIT_KEYS = ['Enter', ',', ';', 'Tab'] as const;
const DEFAULT_SPLIT = /[,;\n\t]+/g;

const normalize = (raw: string): string => raw.trim();

const dedup = (tags: string[]): string[] => {
	const seen = new Set<string>();
	const out: string[] = [];
	for (const t of tags) {
		const key = t.toLowerCase();
		if (seen.has(key)) continue;
		seen.add(key);
		out.push(t);
	}
	return out;
};

export const TagInput = forwardRef<TagInputHandle, TagInputProps>(function TagInput(
	{
		value,
		onChange,
		placeholder = 'Type and press Enter',
		commitKeys = DEFAULT_COMMIT_KEYS,
		splitPattern = DEFAULT_SPLIT,
		id,
		ariaLabel,
		disabled = false,
		maxTags,
		validate,
		autoFocus = false,
	},
	ref,
) {
	const inputRef = useRef<HTMLInputElement>(null);
	const [draft, setDraft] = useState('');

	const commitKeySet = useMemo(() => new Set(commitKeys), [commitKeys]);

	const addTags = useCallback(
		(incoming: string[]): string[] => {
			const cleaned = incoming
				.map(normalize)
				.filter((t) => t.length > 0)
				.filter((t) => (validate ? validate(t) : true));

			if (cleaned.length === 0) return value;

			let next = dedup([...value, ...cleaned]);
			if (typeof maxTags === 'number' && maxTags >= 0 && next.length > maxTags) {
				next = next.slice(0, maxTags);
			}
			if (next.length !== value.length) onChange(next);
			return next;
		},
		[value, onChange, validate, maxTags],
	);

	const removeAt = useCallback(
		(idx: number) => {
			if (idx < 0 || idx >= value.length) return;
			const next = value.slice(0, idx).concat(value.slice(idx + 1));
			onChange(next);
		},
		[value, onChange],
	);

	const commitDraft = useCallback((): boolean => {
		const trimmed = normalize(draft);
		if (trimmed.length === 0) return false;
		addTags([trimmed]);
		setDraft('');
		return true;
	}, [draft, addTags]);

	useImperativeHandle(
		ref,
		() => ({
			flush: () => {
				const trimmed = normalize(draft);
				if (trimmed.length === 0) return value;
				const next = addTags([trimmed]);
				setDraft('');
				return next;
			},
			focus: () => inputRef.current?.focus(),
		}),
		[draft, value, addTags],
	);

	const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
		if (commitKeySet.has(e.key)) {
			// Tab should only commit when there is a pending draft, otherwise let it
			// move focus naturally so the user can leave the field.
			if (e.key === 'Tab' && normalize(draft).length === 0) return;
			const committed = commitDraft();
			if (committed) {
				e.preventDefault();
			} else if (e.key === 'Enter') {
				// Prevent accidental form submission while leaving focus state intact.
				e.preventDefault();
			}
			return;
		}
		if (e.key === 'Backspace' && draft.length === 0 && value.length > 0) {
			e.preventDefault();
			removeAt(value.length - 1);
		}
	};

	const handlePaste = (e: ClipboardEvent<HTMLInputElement>) => {
		const text = e.clipboardData.getData('text');
		if (!text || !splitPattern.test(text)) return;
		e.preventDefault();
		const parts = text.split(splitPattern);
		addTags(parts);
		setDraft('');
	};

	const handleBlur = () => {
		if (normalize(draft).length > 0) commitDraft();
	};

	return (
		<div
			className={[
				'flex min-h-[42px] flex-wrap items-center gap-1.5 rounded-md border bg-white px-2 py-1.5 text-sm transition-colors dark:bg-zinc-900',
				disabled
					? 'cursor-not-allowed border-zinc-200 bg-zinc-50 opacity-70 dark:border-zinc-700 dark:bg-zinc-900/60'
					: 'border-zinc-300 focus-within:border-[#76b900] focus-within:ring-2 focus-within:ring-[#76b900]/30 dark:border-zinc-600',
			].join(' ')}
			onClick={() => inputRef.current?.focus()}
			role="group"
			aria-label={ariaLabel}
		>
			{value.map((tag, idx) => (
				<span
					// Tags must allow duplicates conceptually, but we dedup. Use `tag-idx`
					// as a stable-ish key while still tolerating repeated text content.
					key={`${tag}-${idx}`}
					className="inline-flex max-w-[min(24rem,100%)] items-center gap-1 rounded-md border border-[#76b900]/40 bg-[#76b900]/10 px-2 py-0.5 text-xs font-medium text-[#3f6b00] dark:border-[#76b900]/40 dark:bg-[#76b900]/15 dark:text-[#cdeb86]"
				>
					<Text text={tag} />
					{!disabled && (
						<button
							type="button"
							onClick={(ev) => {
								ev.stopPropagation();
								removeAt(idx);
							}}
							aria-label={`Remove ${tag}`}
							className="-mr-1 flex h-4 w-4 cursor-pointer items-center justify-center rounded-full text-[#3f6b00] transition-colors hover:bg-[#76b900]/25 hover:text-[#2c4d00] dark:text-[#cdeb86] dark:hover:bg-[#76b900]/30"
						>
							<svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden>
								<path
									d="M1.5 1.5L8.5 8.5M8.5 1.5L1.5 8.5"
									stroke="currentColor"
									strokeWidth="1.5"
									strokeLinecap="round"
								/>
							</svg>
						</button>
					)}
				</span>
			))}
			<input
				ref={inputRef}
				id={id}
				type="text"
				value={draft}
				disabled={disabled}
				autoFocus={autoFocus}
				onChange={(e) => setDraft(e.target.value)}
				onKeyDown={handleKeyDown}
				onPaste={handlePaste}
				onBlur={handleBlur}
				placeholder={value.length === 0 ? placeholder : ''}
				aria-label={ariaLabel ?? 'Add tag'}
				className="min-w-[6rem] flex-1 border-0 bg-transparent px-1 py-0.5 text-sm text-zinc-800 outline-none placeholder:text-zinc-400 disabled:cursor-not-allowed dark:text-zinc-100"
			/>
		</div>
	);
});
