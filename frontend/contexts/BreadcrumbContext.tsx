// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import type { BreadcrumbItem } from '@/common/Breadcrumbs';

const PATH_LABELS: Record<string, BreadcrumbItem> = {
	'/chat': { label: 'Chat', href: '/chat' },
	'/terms': { label: 'Terms', href: '/terms' },
	'/analysis': { label: 'Analysis', href: '/analysis' },
	'/exploration': { label: 'Exploration', href: '/exploration' },
	'/data': { label: 'All Data', href: '/data' },
	'/analytics': { label: 'Analytics', href: '/analytics' },
	'/settings': { label: 'Settings', href: '/settings' },
};

const labelForPath = (path: string): BreadcrumbItem | null => {
	const match = Object.entries(PATH_LABELS).find(([prefix]) => path.startsWith(prefix));
	return match ? match[1] : null;
};

type BreadcrumbContextValue = {
	items: BreadcrumbItem[];
	rightSlot: ReactNode;
	setRightSlot: (node: ReactNode) => void;
};

const BreadcrumbContext = createContext<BreadcrumbContextValue>({
	items: [],
	rightSlot: null,
	setRightSlot: () => {},
});

export const BreadcrumbProvider = ({ children }: { children: ReactNode }) => {
	const pathname = usePathname();
	const [prevPath, setPrevPath] = useState<string | null>(null);
	const [items, setItems] = useState<BreadcrumbItem[]>([]);
	const [rightSlot, setRightSlotRaw] = useState<ReactNode>(null);

	const setRightSlot = useCallback((node: ReactNode) => setRightSlotRaw(node), []);

	if (pathname !== prevPath) {
		const current = labelForPath(pathname);
		if (current) {
			const parentCrumb = prevPath ? labelForPath(prevPath) : null;
			if (parentCrumb && parentCrumb.href !== current.href) {
				setItems([parentCrumb, { label: current.label }]);
			} else {
				setItems([{ label: current.label }]);
			}
		}
		setPrevPath(pathname);
	}

	const value = useMemo(
		() => ({ items, rightSlot, setRightSlot }),
		[items, rightSlot, setRightSlot],
	);

	return <BreadcrumbContext.Provider value={value}>{children}</BreadcrumbContext.Provider>;
};

export const useBreadcrumbs = () => useContext(BreadcrumbContext);
