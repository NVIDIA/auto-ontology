// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Popover } from '@/common/Popover';
import { ColorSwatch } from '@/components/settings/ColorSwatch';
import { Size } from '@/enums/button';

export type ColorOption = {
	value: string;
	label: string;
	colorClass: string;
};

type ColorPickerProps = {
	colors: readonly ColorOption[];
	value: string;
	onChange: (next: string) => void;
	fallbackColorClass?: string;
};

export const ColorPicker = ({
	colors,
	value,
	onChange,
	fallbackColorClass = 'bg-zinc-400',
}: ColorPickerProps) => {
	const selectedColorClass =
		colors.find((color) => color.value === value)?.colorClass ?? fallbackColorClass;

	return (
		<Popover
			className="shrink-0"
			panelClassName="w-40 p-2"
			trigger={({ toggle }) => (
				<ColorSwatch
					colorClass={selectedColorClass}
					size={Size.LARGE}
					onClick={toggle}
					aria-label="Pick color"
					title="Pick color"
				/>
			)}
		>
			{({ close }) => (
				<div className="grid grid-cols-3 gap-2">
					{colors.map((color) => (
						<ColorSwatch
							key={color.value}
							colorClass={color.colorClass}
							selected={value === color.value}
							onClick={() => {
								onChange(color.value);
								close();
							}}
							title={color.label}
							aria-label={`Color ${color.label}`}
						/>
					))}
				</div>
			)}
		</Popover>
	);
};
