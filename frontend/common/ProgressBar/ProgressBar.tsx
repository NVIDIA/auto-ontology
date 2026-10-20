// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Size } from '@/enums/button';
import { ProgressBarKind } from '@/enums/progressBar';

const trackSizeClasses: Record<Size, string> = {
	[Size.SMALL]: 'h-1',
	[Size.REGULAR]: 'h-1.5',
	[Size.LARGE]: 'h-2',
};

type ProgressBarBaseProps = {
	size?: Size;
	className?: string;
	'aria-label': string;
	/** Caption to the left of the percent (determinate) or above the bar. */
	label?: string;
	/** When determinate, render the current percent to the right of `label`. */
	showValue?: boolean;
};

export type ProgressBarProps = ProgressBarBaseProps &
	(
		| { kind?: ProgressBarKind.Determinate; value: number }
		| { kind: ProgressBarKind.Indeterminate; value?: never }
	);

const clampPercent = (value: number): number => Math.min(100, Math.max(0, value));

export const ProgressBar = ({
	size = Size.REGULAR,
	className = '',
	label,
	showValue = false,
	'aria-label': ariaLabel,
	...props
}: ProgressBarProps) => {
	const isIndeterminate = props.kind === ProgressBarKind.Indeterminate;
	const percent = isIndeterminate ? 0 : clampPercent(props.value);
	const displayPercent = Math.round(percent);
	const showHeader = Boolean(label) || (showValue && !isIndeterminate);

	return (
		<div className={`w-full ${className}`}>
			{showHeader ? (
				<div className="mb-1.5 flex items-center justify-between gap-3 text-xs text-secondary">
					{label ? <span>{label}</span> : <span />}
					{showValue && !isIndeterminate ? (
						<span className="tabular-nums">{displayPercent}%</span>
					) : null}
				</div>
			) : null}
			<div
				role="progressbar"
				aria-label={ariaLabel}
				aria-valuemin={isIndeterminate ? undefined : 0}
				aria-valuemax={isIndeterminate ? undefined : 100}
				aria-valuenow={isIndeterminate ? undefined : displayPercent}
				aria-valuetext={isIndeterminate ? undefined : `${displayPercent}%`}
				className={`overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700 ${trackSizeClasses[size]}`}
			>
				{isIndeterminate ? (
					<div
						className={`${trackSizeClasses[size]} w-1/3 rounded-full bg-[#76b900] motion-safe:animate-progress-indeterminate`}
					/>
				) : (
					<div
						className={`${trackSizeClasses[size]} rounded-full bg-[#76b900] transition-[width] duration-300 ease-out`}
						style={{ width: `${displayPercent}%` }}
					/>
				)}
			</div>
		</div>
	);
};
