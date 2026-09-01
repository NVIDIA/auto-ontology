// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';

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
		<Button
			theme={ButtonTheme.Icon}
			size={Size.SMALL}
			iconOnly
			type="button"
			onClick={onClick}
			aria-label={label}
			title={label}
		>
			<Icon name={IconName.Link} className="h-3.5 w-3.5" />
		</Button>
	) : null;
