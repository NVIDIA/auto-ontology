// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/components/icons';

/** Small link-style icon button shown next to a count, only rendered when count > 0. */
export const DetailLinkButton = ({
	count,
	onClick,
	label,
}: {
	count: number;
	onClick: () => void;
	label: string;
}) =>
	count > 0 ? (
		<button
			type="button"
			onClick={onClick}
			className="cursor-pointer rounded p-0.5 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:hover:bg-zinc-700"
			aria-label={label}
			title={label}
		>
			<Icon name={IconName.Link} className="h-3.5 w-3.5" />
		</button>
	) : null;
