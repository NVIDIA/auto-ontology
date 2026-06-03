// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';
import { CatalogSubNav } from '@/components/catalog/CatalogSubNav';

export default function CatalogLayout({ children }: { children: ReactNode }) {
	return (
		<div className="flex h-full min-h-0 flex-col bg-zinc-50 dark:bg-zinc-950">
			<CatalogSubNav />
			<div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
		</div>
	);
}
