// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { AUTO_GENERATED_LABEL } from '@/constants/tags';
import type { TagAuthor } from '@/types/tags';

/**
 * A person a rule records, with the initial the rest of the app draws one as.
 *
 * Takes the resolved account rather than the rule, because a rule records two —
 * whoever saved it and whoever last renamed it — and they read identically. The
 * label above each is what says which is which.
 *
 * "Auto Generated" when the id resolved to nobody, as everywhere else in the
 * app. For an author that means the account has since been deleted, which these
 * ids outlive because they are not foreign keys; a rule always has one, since
 * the backend refuses a create it cannot attribute. For an editor it also
 * covers a rename that carried no identity to record, which nothing on the
 * deployed path does.
 */
export const RuleAuthor = ({ author }: { author?: TagAuthor | null }) => {
	const name = author?.name || author?.email || '';
	const label = name || AUTO_GENERATED_LABEL;

	return (
		<span className="flex min-w-0 items-center gap-2">
			<span
				aria-hidden="true"
				className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${name === '' ? 'bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400' : 'bg-[#76b900] text-white'}`}
			>
				{label.charAt(0).toUpperCase()}
			</span>
			<span className="min-w-0 truncate text-xs text-zinc-600 dark:text-zinc-300">
				{label}
			</span>
		</span>
	);
};
