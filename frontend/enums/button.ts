// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export enum ButtonTheme {
	Primary = 'primary',
	Secondary = 'secondary',
	Outline = 'outline',
	Danger = 'danger',
	DangerOutline = 'danger-outline',
	/** Pale bordered danger button (subtler than DangerOutline's solid hover-invert). */
	DangerSubtle = 'danger-subtle',
	/** Low-emphasis text button: no fill or border, brand-accent tint on hover. */
	Minimal = 'minimal',
	/** Icon-only button with brand-accent hover (green tint). */
	Icon = 'icon',
	/** Icon-only button with a neutral gray hover, for generic utility controls (close, dots, expand/collapse…). */
	IconNeutral = 'icon-neutral',
	/** Icon-only button with a red hover, for destructive row actions. */
	IconDanger = 'icon-danger',
	/** Light bordered/tinted brand-accent button (text stays green, background only tints on hover). */
	Soft = 'soft',
}

export enum Size {
	SMALL = 'small',
	REGULAR = 'regular',
	LARGE = 'large',
}

export enum SelectButtonTheme {
	Avatar = 'avatar',
	ListItem = 'list-item',
	/** Row that navigates elsewhere: label on the left, link icon on the right. */
	ListItemLink = 'list-item-link',
	/** Row with a title over a caption, leaving room for a trailing action. */
	ListItemTwoLine = 'list-item-two-line',
	/** Field showing the current selection and opening a dropdown when clicked. */
	SelectField = 'select-field',
	/** One option of a mode switcher sitting inside a shared container. */
	Switcher = 'switcher',
}
