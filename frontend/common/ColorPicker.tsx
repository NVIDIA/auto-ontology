// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState } from 'react';

export type ColorOption = {
	value: string;
	label: string;
	swatchClassName: string;
};

type ColorPickerProps = {
	colors: readonly ColorOption[];
	value: string;
	onChange: (next: string) => void;
	fallbackSwatchClassName?: string;
};

export const ColorPicker = ({
	colors,
	value,
	onChange,
	fallbackSwatchClassName = 'bg-zinc-400',
}: ColorPickerProps) => {
	const [open, setOpen] = useState(false);
	const ref = useRef<HTMLDivElement>(null);
	const selectedSwatchClassName =
		colors.find((color) => color.value === value)?.swatchClassName ?? fallbackSwatchClassName;

	useEffect(() => {
		if (!open) return;
		const handleClick = (event: MouseEvent) => {
			if (ref.current && !ref.current.contains(event.target as Node)) {
				setOpen(false);
			}
		};
		document.addEventListener('mousedown', handleClick);
		return () => document.removeEventListener('mousedown', handleClick);
	}, [open]);

	return (
		<div ref={ref} className="relative shrink-0">
			<button
				type="button"
				onClick={() => setOpen((prev) => !prev)}
				aria-label="Pick color"
				title="Pick color"
				className={`h-9 w-9 cursor-pointer rounded-full border-2 transition ${selectedSwatchClassName} border-white/80 ring-1 ring-zinc-300/80 hover:scale-105 dark:border-zinc-900 dark:ring-zinc-700`}
			/>
			{open ? (
				<div className="absolute right-0 top-full z-20 mt-2 w-40 rounded-lg border border-zinc-200 bg-white p-2 shadow-lg dark:border-zinc-700 dark:bg-zinc-800">
					<div className="grid grid-cols-3 gap-2">
						{colors.map((color) => {
							const selected = value === color.value;
							return (
								<button
									key={color.value}
									type="button"
									onClick={() => {
										onChange(color.value);
										setOpen(false);
									}}
									title={color.label}
									aria-label={`Color ${color.label}`}
									className={`h-8 w-8 cursor-pointer rounded-full border-2 ${color.swatchClassName} ${
										selected
											? 'border-zinc-900 ring-2 ring-[#76b900]/60 dark:border-zinc-100'
											: 'border-white/80 ring-1 ring-zinc-300/80 dark:border-zinc-900 dark:ring-zinc-700'
									}`}
								/>
							);
						})}
					</div>
				</div>
			) : null}
		</div>
	);
};
