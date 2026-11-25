// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Icon, IconName } from '@/common/icons';

/** Menu icons the design ships, each in default, hover and selected versions. */
type DesignIconName =
	| 'chat'
	| 'discovery'
	| 'glossary'
	| 'observability'
	| 'exploration'
	| 'data'
	| 'settings';

type NavItem = {
	/** Icon from the design. Items the design has no icon for use `icon` instead. */
	designIcon?: DesignIconName;
	icon?: IconName;
	href: string;
	label: string;
	adminOnly?: boolean;
};

const navItems: NavItem[] = [
	{ designIcon: 'chat', href: '/chat', label: 'Chat' },
	{ designIcon: 'discovery', href: '/discovery', label: 'Discovery' },
	{ designIcon: 'glossary', href: '/terms', label: 'Glossary' },
	{ designIcon: 'observability', href: '/analysis', label: 'Analysis' },
	{ designIcon: 'exploration', href: '/exploration', label: 'Exploration' },
	{ designIcon: 'data', href: '/data', label: 'Data' },
	{ icon: IconName.ChartLine, href: '/analytics', label: 'Analytics', adminOnly: true },
	// Settings is admin-only and now also contains Users and Single Sign-On
	// (see app/settings/*). Viewers don't see it.
	{
		designIcon: 'settings',
		href: '/settings',
		label: 'Settings',
		adminOnly: true,
	},
];

const TILE = 'relative block size-9 max-w-none shrink-0';

const TILE_CLIP = 'block w-9 shrink-0';

// Open, the label's color carries on from the icon tile. The tile's own rounded
// right corners would leave a notch where the two meet, so these two corner
// pieces fill just that gap, outside the tile's curve, with the row's color
// (`--row`; `--r` is the tile's corner radius, which differs per state).
const NOTCH =
	'pointer-events-none absolute hidden size-[var(--r)] [left:calc(var(--r)*-1)] group-hover/rail:block';
const NOTCH_TOP =
	'top-0 [background:radial-gradient(circle_at_0_100%,transparent_var(--r),var(--row)_var(--r))]';
const NOTCH_BOTTOM =
	'bottom-0 [background:radial-gradient(circle_at_0_0,transparent_var(--r),var(--row)_var(--r))]';

type Theme = 'light' | 'dark';

// The icons are static, fixed-size files, so a plain <img> is all they need.
/* eslint-disable @next/next/no-img-element */

// The design's hover icon for Chat comes as loose parts rather than one file:
// a pale tile, the bubble outline and three green lines.
const ChatHoverIcon = ({ theme }: { theme: Theme }) => {
	const dark = theme === 'dark';
	const line = `absolute left-3 h-px rounded-[2px] ${dark ? 'bg-[#8ad21a]' : 'bg-[#76b900]'}`;

	return (
		<span
			className={`${TILE} hidden rounded group-hover:block group-hover/rail:rounded-r-none ${
				dark ? 'bg-[rgba(138,210,26,0.12)]' : 'bg-[#f1f8e5]'
			}`}
		>
			<img
				src={dark ? '/nav/dark/chat-hover-bubble.svg' : '/nav/chat-hover-bubble.svg'}
				alt=""
				className="absolute left-[7.5px] top-[9px] h-[20.3px] w-[21px] max-w-none"
			/>
			<span className={`${line} top-[14px] w-3`} />
			<span className={`${line} top-[17px] w-[11px]`} />
			<span className={`${line} top-5 w-[7px]`} />
		</span>
	);
};

// One theme's icon states. Which theme shows is the page's color scheme,
// decided in CSS by the wrapper in `DesignIcon`, so nothing waits on JavaScript.
const IconStates = ({
	name,
	active,
	theme,
}: {
	name: DesignIconName;
	active: boolean;
	theme: Theme;
}) => {
	const dir = theme === 'dark' ? '/nav/dark' : '/nav';

	if (active) {
		return (
			<img
				src={`${dir}/${name}-selected.svg`}
				alt=""
				width={36}
				height={36}
				className={TILE}
			/>
		);
	}

	return (
		<>
			<img
				src={`${dir}/${name}-default.svg`}
				alt=""
				width={36}
				height={36}
				className={`${TILE} group-hover:hidden`}
			/>
			{name === 'chat' ? (
				<ChatHoverIcon theme={theme} />
			) : (
				<img
					src={`${dir}/${name}-hover.svg`}
					alt=""
					width={36}
					height={36}
					className={`${TILE} hidden group-hover:block`}
				/>
			)}
		</>
	);
};
/* eslint-enable @next/next/no-img-element */

const DesignIcon = ({ name, active }: { name: DesignIconName; active: boolean }) => (
	<>
		<span className={`${TILE_CLIP} dark:hidden`}>
			<IconStates name={name} active={active} theme="light" />
		</span>
		<span className={`${TILE_CLIP} hidden dark:block`}>
			<IconStates name={name} active={active} theme="dark" />
		</span>
	</>
);

// The rail opens on hover with CSS alone (`group/rail`), not state, and with no
// animation: the width, the row highlight, the label and the icon's hover then
// all change in the same frame. An animated width would show the icon's hover
// first and the rest of the row only as it caught up.
export const NavRail = ({ isAdmin = false }: { isAdmin?: boolean }) => {
	const pathname = usePathname();
	const visibleItems = navItems.filter((item) => !item.adminOnly || isAdmin);

	const renderItem = (item: NavItem) => {
		const isActive = pathname.startsWith(item.href);

		// Open, the row's color runs on past the icon tile behind the label: pale
		// green on hover, a stronger green for the current page. The tile carries
		// its own color, so the two meet without doubling up.
		const labelColor = isActive
			? '[--r:8px] [--row:#ddedbf] bg-[#ddedbf] text-[#76b900] dark:[--row:rgba(138,210,26,0.28)] dark:bg-[rgba(138,210,26,0.28)] dark:text-[#8ad21a]'
			: '[--r:4px] [--row:transparent] text-[#2b3036] group-hover:[--row:#f1f8e5] group-hover:bg-[#f1f8e5] group-hover:text-[#76b900] dark:text-[#f0f0f3] dark:group-hover:[--row:rgba(138,210,26,0.12)] dark:group-hover:bg-[rgba(138,210,26,0.12)] dark:group-hover:text-[#8ad21a]';

		const fallbackIconColor = isActive
			? 'rounded-lg group-hover/rail:rounded-r-none bg-[#ddedbf] text-[#76b900] dark:bg-[rgba(138,210,26,0.28)] dark:text-[#8ad21a]'
			: 'rounded group-hover/rail:rounded-r-none text-[#2b3036] group-hover:bg-[#f1f8e5] group-hover:text-[#76b900] dark:text-[#f0f0f3] dark:group-hover:bg-[rgba(138,210,26,0.12)] dark:group-hover:text-[#8ad21a]';

		return (
			<Link
				key={item.href}
				href={item.href}
				className="group flex h-9 w-full shrink-0 items-center overflow-hidden rounded-lg"
			>
				{item.designIcon ? (
					<DesignIcon name={item.designIcon} active={isActive} />
				) : (
					<span className={`${TILE_CLIP} h-9 ${fallbackIconColor}`}>
						<span className="flex size-9 items-center justify-center">
							{item.icon ? (
								<Icon name={item.icon} className="h-5 w-5 shrink-0" />
							) : null}
						</span>
					</span>
				)}
				<span
					className={`relative flex h-9 min-w-0 flex-1 items-center rounded-r-lg pr-3 text-sm leading-5 whitespace-nowrap ${labelColor}`}
				>
					<span aria-hidden className={`${NOTCH} ${NOTCH_TOP}`} />
					<span aria-hidden className={`${NOTCH} ${NOTCH_BOTTOM}`} />
					<span className="truncate">{item.label}</span>
				</span>
			</Link>
		);
	};

	return (
		<nav className="relative h-full w-[60px] shrink-0 border-r border-[#e9eaea] bg-white font-sf dark:border-[#2e2e33] dark:bg-[#18181b]">
			<div className="group/rail absolute inset-y-0 left-0 flex w-full flex-col gap-3 overflow-hidden bg-white px-3 py-4 hover:z-30 hover:w-[148px] hover:shadow-[0_2px_6px_rgba(80,80,80,0.25)] dark:bg-[#18181b] dark:hover:shadow-[0_2px_6px_rgba(0,0,0,0.4)]">
				{visibleItems.map(renderItem)}
			</div>
		</nav>
	);
};
