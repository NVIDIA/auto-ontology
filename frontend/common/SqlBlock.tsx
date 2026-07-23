// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useState } from 'react';

type CopyButtonProps = {
	text: string;
};

const CopyButton = ({ text }: CopyButtonProps) => {
	const [copied, setCopied] = useState(false);

	const handleCopy = useCallback(() => {
		navigator.clipboard.writeText(text).then(() => {
			setCopied(true);
			setTimeout(() => setCopied(false), 2000);
		});
	}, [text]);

	return (
		<button
			type="button"
			onClick={handleCopy}
			className="absolute top-2 right-2 rounded bg-zinc-700 px-2 py-1 text-xs text-zinc-300 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-zinc-600"
		>
			{copied ? 'Copied!' : 'Copy'}
		</button>
	);
};

export type SqlBlockProps = {
	sql: string;
	label?: string;
	className?: string;
};

export const SqlBlock = ({ sql, label = 'SQL', className }: SqlBlockProps) => (
	<div
		className={`group relative overflow-hidden rounded-lg bg-zinc-900 dark:bg-zinc-950 ${className ?? ''}`}
	>
		<div className="flex items-center justify-between border-b border-zinc-700 px-3 py-1.5">
			<span className="text-xs font-medium text-zinc-400">{label}</span>
			<CopyButton text={sql} />
		</div>
		<pre className="p-3 text-xs leading-relaxed whitespace-pre-wrap break-words text-[#76b900]">
			<code>{sql}</code>
		</pre>
	</div>
);

export type SqlEditorProps = {
	value: string;
	onChange: (value: string) => void;
	label?: string;
	placeholder?: string;
	rows?: number;
	className?: string;
};

export const SqlEditor = ({
	value,
	onChange,
	label = 'SQL',
	placeholder = 'SELECT ...',
	rows = 8,
	className,
}: SqlEditorProps) => (
	<div
		className={`overflow-hidden rounded-lg bg-zinc-900 transition-colors focus-within:ring-2 focus-within:ring-[#76b900]/30 dark:bg-zinc-950 ${className ?? ''}`}
	>
		<div className="flex items-center justify-between border-b border-zinc-700 px-3 py-1.5">
			<span className="text-xs font-medium text-zinc-400">{label}</span>
		</div>
		<textarea
			value={value}
			onChange={(e) => onChange(e.target.value)}
			placeholder={placeholder}
			rows={rows}
			spellCheck={false}
			aria-label={label}
			className="block w-full resize-y bg-transparent p-3 font-mono text-xs leading-relaxed text-[#76b900] outline-none placeholder:text-zinc-600"
		/>
	</div>
);
