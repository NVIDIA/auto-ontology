// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

import { Button } from '@/common/Button';
import { Icon, IconName } from '@/common/icons';
import { ButtonTheme, Size } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';

export type EmptyStateAction = {
	label: string;
	icon?: IconName;
	onClick: () => void;
	theme?: ButtonTheme;
	disabled?: boolean;
};

export type EmptyStateProps = {
	title: ReactNode;
	description?: ReactNode;
	/** Shorthand for a muted glyph above the title. Ignored when `illustration` is set. */
	icon?: IconName;
	/** Custom artwork above the title, e.g. one of `Placeholders.*`. */
	illustration?: ReactNode;
	action?: EmptyStateAction;
	variant?: EmptyStateVariant;
	className?: string;
};

type VariantStyle = {
	container: string;
	icon: string;
	title: string;
	description: string;
};

const variantStyles: Record<EmptyStateVariant, VariantStyle> = {
	[EmptyStateVariant.Dashed]: {
		container:
			'min-h-[40dvh] rounded-2xl border border-dashed border-zinc-300/90 bg-white/60 p-12 dark:border-zinc-700 dark:bg-zinc-950/40',
		icon: 'h-8 w-8',
		title: 'text-sm font-semibold text-body',
		description: 'text-xs text-secondary',
	},
	[EmptyStateVariant.Borderless]: {
		container: 'h-full flex-1 p-8',
		icon: 'h-8 w-8',
		title: 'text-sm font-semibold text-body',
		description: 'text-xs text-secondary',
	},
	[EmptyStateVariant.Inline]: {
		container: 'px-6 py-8',
		icon: 'h-6 w-6',
		title: 'text-xs font-semibold text-secondary',
		description: 'text-xs text-secondary',
	},
	[EmptyStateVariant.Welcome]: {
		container: 'h-full flex-1 px-4 py-10',
		icon: 'h-8 w-8',
		title: 'text-xl font-semibold text-heading',
		description: 'text-sm text-secondary',
	},
};

export const EmptyState = ({
	title,
	description,
	icon,
	illustration,
	action,
	variant = EmptyStateVariant.Dashed,
	className = '',
}: EmptyStateProps) => {
	const styles = variantStyles[variant];
	const iconPosition = action?.icon ? 'left' : undefined;
	const actionContent = action && (
		<>
			{action.icon && <Icon name={action.icon} className="h-4 w-4" />}
			{action.label}
		</>
	);

	return (
		<div
			className={[
				'flex w-full flex-col items-center justify-center gap-3 text-center',
				styles.container,
				className,
			]
				.filter(Boolean)
				.join(' ')}
		>
			{illustration ??
				(icon && <Icon name={icon} className={`${styles.icon} text-disabled `} />)}
			<p className={styles.title}>{title}</p>
			{description && <p className={`max-w-sm ${styles.description}`}>{description}</p>}
			{action && (
				<Button
					theme={action.theme ?? ButtonTheme.Primary}
					size={Size.REGULAR}
					type="button"
					onClick={action.onClick}
					disabled={action.disabled}
					iconPosition={iconPosition}
					shadow
				>
					{actionContent}
				</Button>
			)}
		</div>
	);
};
