// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { ReactNode } from 'react';
import { ThemeProvider } from '@kui/foundations-react';

/**
 * Scopes the Kaizen UI Foundations design system to the catalog subtree.
 *
 * `theme="system"` makes the provider follow the OS colour-scheme preference and
 * apply the matching `.nv-dark` / `.nv-light` class, which is what Kaizen's
 * design tokens (defined in `@kui/foundations-react/base.css`) key off. Keeping
 * the provider local (rather than global on <html>) means the rest of the app
 * is untouched by Kaizen theming.
 */
export const CatalogThemeProvider = ({ children }: { children: ReactNode }) => (
	<ThemeProvider
		theme="system"
		density="standard"
		motion="system"
		className="flex h-full min-h-0 flex-col"
	>
		{children}
	</ThemeProvider>
);
