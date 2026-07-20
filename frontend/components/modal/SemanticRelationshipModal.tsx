// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/components/icons';
import { Modal } from './Modal';

/** Minimal Term reference — decoupled from any specific page's node/row shape. */
export type SemanticRelationshipModalTerm = {
	id: string;
	name: string;
};

type SemanticRelationshipModalProps = {
	sourceTerm: SemanticRelationshipModalTerm | null;
	targetTerm: SemanticRelationshipModalTerm | null;
	onClose: () => void;
	onView: (termId: string) => void;
};

/** Generic modal showing the two Terms joined by a semantic-graph edge. */
export const SemanticRelationshipModal = ({
	sourceTerm,
	targetTerm,
	onClose,
	onView,
}: SemanticRelationshipModalProps) => {
	const open = sourceTerm != null && targetTerm != null;

	return (
		<Modal open={open} onClose={onClose} className="w-full max-w-2xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon name={IconName.Connection} className="h-5 w-5 shrink-0 text-[#76b900]" />
					<h2 className="truncate text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{sourceTerm?.name} (Term) &lt;&gt; {targetTerm?.name} (Term)
					</h2>
				</div>
				<button
					type="button"
					onClick={onClose}
					className="cursor-pointer rounded p-1 text-lg text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-700 dark:hover:text-zinc-200"
					aria-label="Close term relationship"
				>
					×
				</button>
			</header>
			<div className="p-5">
				<div className="grid grid-cols-2 overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700">
					{[sourceTerm, targetTerm].map((term, index) => (
						<div
							key={term?.id ?? index}
							className={
								index === 0 ? 'border-r border-zinc-200 dark:border-zinc-700' : ''
							}
						>
							<div className="border-b border-zinc-200 bg-zinc-50 px-4 py-2.5 dark:border-zinc-700 dark:bg-zinc-800/60">
								<span className="truncate text-xs font-semibold text-zinc-500 dark:text-zinc-400">
									{term?.name}
								</span>
							</div>
							<button
								type="button"
								onClick={() => term != null && onView(term.id)}
								className="flex w-full cursor-pointer items-center justify-between gap-2 px-4 py-3 text-left text-sm text-zinc-700 transition-colors hover:bg-zinc-50 dark:text-zinc-300 dark:hover:bg-zinc-800/40"
							>
								<span className="flex min-w-0 items-center gap-1.5">
									<Icon
										name={IconName.Terms}
										className="h-3.5 w-3.5 shrink-0 text-[#76b900]"
									/>
									<span className="truncate">{term?.name}</span>
								</span>
								<Icon
									name={IconName.ExternalLink}
									className="h-4 w-4 shrink-0 text-zinc-400"
								/>
							</button>
						</div>
					))}
				</div>
			</div>
		</Modal>
	);
};
