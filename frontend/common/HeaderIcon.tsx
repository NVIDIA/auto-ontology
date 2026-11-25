// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The top bar's icons, as the design ships them: one file per color scheme,
 * with the colors baked in. Both are rendered and CSS shows the one that
 * matches the scheme, so the icon follows `prefers-color-scheme` like the
 * rest of the app without the files being edited.
 */
export type HeaderIconName = 'arrows' | 'search' | 'search-hover';

// Plain <img>: static, fixed-size icons that need no optimization.
/* eslint-disable @next/next/no-img-element */
export const HeaderIcon = ({ name }: { name: HeaderIconName }) => (
	<>
		<img
			src={`/header/${name}-light.svg`}
			alt=""
			width={24}
			height={24}
			className="block size-6 shrink-0 dark:hidden"
		/>
		<img
			src={`/header/${name}-dark.svg`}
			alt=""
			width={24}
			height={24}
			className="hidden size-6 shrink-0 dark:block"
		/>
	</>
);
/* eslint-enable @next/next/no-img-element */
