// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState, type ReactNode } from 'react';

export type PopoverMenuItem = {
	label: string;
	icon?: ReactNode;
	onClick: () => void;
	danger?: boolean;
};

type PopoverMenuProps = {
	items: PopoverMenuItem[];
	trigger: (props: { open: boolean; toggle: (e: React.MouseEvent) => void }) => ReactNode;
	className?: string;
};

export const PopoverMenu = ({ items, trigger, className = '' }: PopoverMenuProps) => {
	const [open, setOpen] = useState(false);
	const ref = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (!open) return;
		const handleClick = (e: MouseEvent) => {
			if (ref.current && !ref.current.contains(e.target as Node)) {
				setOpen(false);
			}
		};
		document.addEventListener('mousedown', handleClick);
		return () => document.removeEventListener('mousedown', handleClick);
	}, [open]);

	const toggle = (e: React.MouseEvent) => {
		e.stopPropagation();
		setOpen((o) => !o);
	};

	return (
		<div ref={ref} className={className}>
			{trigger({ open, toggle })}
			{open && (
				<div className="absolute right-0 top-full z-30 mt-1 w-32 rounded-lg border border-zinc-200 bg-white py-1 shadow-lg dark:border-zinc-700 dark:bg-zinc-800">
					{items.map((item) => (
						<button
							key={item.label}
							type="button"
							onClick={() => {
								setOpen(false);
								item.onClick();
							}}
							className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm ${
								item.danger
									? 'text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/40'
									: 'text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-700'
							}`}
						>
							{item.icon}
							{item.label}
						</button>
					))}
				</div>
			)}
		</div>
	);
};
