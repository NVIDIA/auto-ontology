// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState, type ButtonHTMLAttributes } from 'react';

import { Icon, IconName } from '@/common/icons';

const COPIED_RESET_MS = 2000;

/**
 * The async Clipboard API is only available in secure contexts and rejects when
 * the document is not focused, so fall back to a throwaway textarea selection.
 */
const copyToClipboard = async (text: string): Promise<boolean> => {
	try {
		await navigator.clipboard.writeText(text);
		return true;
	} catch {
		const textArea = document.createElement('textarea');
		textArea.value = text;
		textArea.style.position = 'fixed'; // prevent scrolling to the bottom
		textArea.style.opacity = '0';
		document.body.appendChild(textArea);
		textArea.select();

		try {
			return document.execCommand('copy');
		} catch {
			return false;
		} finally {
			textArea.remove();
		}
	}
};

export type CopyButtonProps = Omit<
	ButtonHTMLAttributes<HTMLButtonElement>,
	'children' | 'onClick'
> & {
	text: string;
};

export const CopyButton = ({ text, className = '', ...props }: CopyButtonProps) => {
	const [copied, setCopied] = useState(false);
	const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

	useEffect(
		() => () => {
			if (resetTimer.current) clearTimeout(resetTimer.current);
		},
		[],
	);

	const handleCopy = useCallback(() => {
		void copyToClipboard(text).then((succeeded) => {
			if (!succeeded) return;
			setCopied(true);
			if (resetTimer.current) clearTimeout(resetTimer.current);
			resetTimer.current = setTimeout(() => setCopied(false), COPIED_RESET_MS);
		});
	}, [text]);

	return (
		<button
			{...props}
			type="button"
			onClick={handleCopy}
			aria-label={copied ? 'Copied to clipboard' : 'Copy to clipboard'}
			className={`rounded p-1 text-zinc-400 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-zinc-700 hover:text-zinc-200 focus-visible:opacity-100 disabled:cursor-not-allowed ${className}`}
		>
			<Icon
				name={copied ? IconName.Check : IconName.Copy}
				className={`h-4 w-4 ${copied ? 'text-[#76b900]' : ''}`}
			/>
		</button>
	);
};
