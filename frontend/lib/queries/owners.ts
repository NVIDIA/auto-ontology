// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { queryOptions, type QueryClient } from '@tanstack/react-query';

import { ownersApi } from '@/api/owners';
import type { Owner } from '@/types/owners';

const ALL_KEY = ['owners'] as const;

const readOwners = async (): Promise<Owner[]> => {
	const res = await ownersApi.list();
	if (res.error) throw new Error(res.message ?? 'Failed to load owners.');
	return res.data ?? [];
};

export const ownerQueries = {
	all: () =>
		queryOptions({
			queryKey: ALL_KEY,
			queryFn: readOwners,
			staleTime: 60_000,
		}),
};

/**
 * The owner directory for a detail page. The page only uses it to label the
 * owner, so a failed directory request yields an empty list instead of
 * rejecting the object's own load.
 */
export const fetchOwnerDirectory = (queryClient: QueryClient): Promise<Owner[]> =>
	queryClient.fetchQuery(ownerQueries.all()).catch(() => []);

/** Drop the cached directory so an open picker refetches users and groups. */
export const invalidateOwners = (queryClient: QueryClient): Promise<void> =>
	queryClient.invalidateQueries({ queryKey: ALL_KEY });
