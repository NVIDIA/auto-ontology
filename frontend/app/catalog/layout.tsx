// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';
import { CatalogSubNav } from '@/components/catalog/CatalogSubNav';
import { CatalogThemeProvider } from '@/components/catalog/CatalogThemeProvider';

export default function CatalogLayout({ children }: { children: ReactNode }) {
	return (
		<CatalogThemeProvider>
			<div className="flex h-full min-h-0 flex-col bg-[var(--background-color-surface-sunken)]">
				<CatalogSubNav />
				<div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
			</div>
		</CatalogThemeProvider>
	);
}
