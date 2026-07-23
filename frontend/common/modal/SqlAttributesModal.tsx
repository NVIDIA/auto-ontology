// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

import { termsApi } from '@/api/terms';
import { Icon, IconName } from '@/common/icons';
import type { SqlAttribute } from '@/types/terms';
import { SqlBlock } from '@/common/SqlBlock';
import { Modal } from './Modal';

/** Minimal Term reference — decoupled from any specific page's node/row shape. */
export type SqlAttributesModalTerm = {
	id: string;
	name: string;
};

type SqlAttributesModalProps = {
	term: SqlAttributesModalTerm | null;
	onClose: () => void;
};

// Description text is width-constrained by the caller and truncated; hovering
// reveals the full text in a floating popover instead of relying on the
// native `title` tooltip.
const TruncatedDescription = ({ text }: { text: string | null }) => {
	const [hovered, setHovered] = useState(false);
	const value = text?.trim() ?? '';

	if (value === '') {
		return <span className="italic text-zinc-400 dark:text-zinc-500">No Description</span>;
	}

	return (
		<span
			className="relative inline-block max-w-full"
			onMouseEnter={() => setHovered(true)}
			onMouseLeave={() => setHovered(false)}
		>
			<span className="block truncate">{value}</span>
			{hovered && (
				<span className="absolute left-0 top-full z-40 mt-1 block w-72 max-w-[min(22rem,90vw)] whitespace-normal rounded-lg border border-zinc-200 bg-white p-2.5 text-xs leading-5 text-zinc-600 shadow-xl dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
					{value}
				</span>
			)}
		</span>
	);
};

/** Generic modal listing a Term's SQL Attributes. Reusable from any page that has a term id. */
export const SqlAttributesModal = ({ term, onClose }: SqlAttributesModalProps) => {
	const [attributes, setAttributes] = useState<SqlAttribute[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		if (term == null) return undefined;
		let cancelled = false;

		const load = async () => {
			setLoading(true);
			setError(null);
			const response = await termsApi.getSqlAttributes(term.id);
			if (cancelled) return;
			if (response.error) {
				setError(response.message ?? 'Failed to load SQL attributes');
			} else {
				setAttributes(response.data ?? []);
			}
			setLoading(false);
		};

		void load();
		return () => {
			cancelled = true;
		};
	}, [term]);

	return (
		<Modal open={term != null} onClose={onClose} className="w-full max-w-2xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon name={IconName.Link} className="h-5 w-5 shrink-0 text-[#76b900]" />
					<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{term?.name} — SQL Attributes ({attributes.length})
					</h2>
				</div>
				<button
					type="button"
					onClick={onClose}
					className="cursor-pointer rounded p-1 text-lg text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
					aria-label="Close SQL attributes"
				>
					×
				</button>
			</header>
			<div className="max-h-[70dvh] overflow-y-auto p-5">
				{loading ? (
					<div className="flex h-32 items-center justify-center">
						<div
							className="h-8 w-8 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900]"
							role="status"
							aria-label="Loading SQL attributes"
						/>
					</div>
				) : error != null ? (
					<p className="text-sm text-red-600 dark:text-red-300">{error}</p>
				) : attributes.length === 0 ? (
					<p className="text-sm italic text-zinc-500 dark:text-zinc-400">
						No SQL attributes
					</p>
				) : (
					<ul className="space-y-4">
						{attributes.map((attr) => (
							<li
								key={attr.id}
								className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700"
							>
								<h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
									{attr.name}
								</h3>
								<div className="mt-1 max-w-md text-xs text-zinc-500 dark:text-zinc-400">
									<TruncatedDescription text={attr.description} />
								</div>
								<SqlBlock
									className="mt-2"
									sql={attr.expression || attr.sql || ''}
									label="SQL"
								/>
							</li>
						))}
					</ul>
				)}
			</div>
		</Modal>
	);
};
