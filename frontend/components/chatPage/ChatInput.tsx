// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useRef, useState, type KeyboardEvent, type FormEvent } from 'react';
import { Icon, IconName } from '@/common/icons';

type ChatInputProps = {
	/** Resolves false when the question was refused, so the text can be restored. */
	onSend: (text: string) => Promise<boolean>;
	onStop: () => void;
	isLoading: boolean;
};

export const ChatInput = ({ onSend, onStop, isLoading }: ChatInputProps) => {
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	// Track whether the textarea currently has any non-whitespace text so the
	// Send button can be disabled/greyed-out when there's nothing to send.
	// We intentionally keep the textarea uncontrolled (perf + caret behaviour)
	// and only mirror the empty/non-empty flag into React state.
	const [hasText, setHasText] = useState(false);

	const resetHeight = () => {
		const el = textareaRef.current;
		if (!el) return;
		el.style.height = 'auto';
		el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
	};

	const handleInput = useCallback((e: FormEvent<HTMLTextAreaElement>) => {
		resetHeight();
		setHasText(e.currentTarget.value.trim().length > 0);
	}, []);

	const handleSubmit = useCallback(
		async (e?: FormEvent) => {
			e?.preventDefault();
			const el = textareaRef.current;
			if (!el) return;
			const text = el.value.trim();
			if (!text || isLoading) return;

			// Clear straight away so sending feels immediate, then put the text
			// back if the backend refused it — a rejected question is never added
			// to the transcript, so discarding it would lose it entirely.
			el.value = '';
			el.style.height = 'auto';
			setHasText(false);

			const accepted = await onSend(text);
			if (accepted) return;

			const target = textareaRef.current;
			if (!target || target.value.trim()) return;
			target.value = text;
			resetHeight();
			setHasText(true);
		},
		[onSend, isLoading],
	);

	const handleKeyDown = useCallback(
		(e: KeyboardEvent<HTMLTextAreaElement>) => {
			if (e.key === 'Enter' && !e.shiftKey) {
				e.preventDefault();
				handleSubmit();
			}
		},
		[handleSubmit],
	);

	return (
		<form
			onSubmit={handleSubmit}
			className="border-t border-zinc-200 bg-white px-4 py-3 dark:border-zinc-700 dark:bg-zinc-900"
		>
			<div className="mx-auto flex max-w-3xl items-end gap-2">
				<textarea
					ref={textareaRef}
					rows={1}
					placeholder="Ask a question…"
					onInput={handleInput}
					onKeyDown={handleKeyDown}
					className="flex-1 resize-none rounded-xl border border-zinc-300 bg-zinc-50 px-4 py-2.5 text-sm text-zinc-900 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-1 focus:ring-[#76b900] dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100 dark:placeholder:text-zinc-500"
				/>

				{isLoading ? (
					<button
						type="button"
						onClick={onStop}
						className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-red-500 text-white transition-colors hover:bg-red-600"
						aria-label="Stop generation"
					>
						<Icon name={IconName.Stop} className="h-4 w-4" />
					</button>
				) : (
					<button
						type="submit"
						disabled={!hasText}
						className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#76b900] text-white transition-colors hover:bg-[#5e9400] disabled:cursor-not-allowed disabled:bg-zinc-300 disabled:text-zinc-500 disabled:hover:bg-zinc-300 dark:disabled:bg-zinc-700 dark:disabled:text-zinc-400 dark:disabled:hover:bg-zinc-700"
						aria-label="Send message"
					>
						<Icon name={IconName.Send} className="h-4 w-4" />
					</button>
				)}
			</div>
		</form>
	);
};
