// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export enum EmptyStateVariant {
	/** Dashed placeholder card, for the main content area of a list page. */
	Dashed = 'dashed',
	/** Block without a card frame, centred in whatever space it is given. */
	Borderless = 'borderless',
	/** Tight spacing and smaller type, for sidebars, tables, modals and inline panels. */
	Inline = 'inline',
	/** Large type, for a landing screen where the empty state is the main content. */
	Welcome = 'welcome',
}
