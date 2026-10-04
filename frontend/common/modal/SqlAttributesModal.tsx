// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback } from 'react';

import { termsApi } from '@/api/terms';
import { Icon, IconName } from '@/common/icons';
import { Button } from '@/common/Button';
import { EmptyState } from '@/common/EmptyState';
import { InfiniteScroll } from '@/common/InfiniteScroll';
import { Size, ButtonTheme } from '@/enums/button';
import { EmptyStateVariant } from '@/enums/emptyState';
import { useInfiniteList } from '@/hooks/useInfiniteList';
import { SkeletonSqlBlocks } from '@/common/Skeleton';
import { SqlBlock } from '@/common/SqlBlock';
import { Text } from '@/common/Text';
import { TextVariant } from '@/enums/text';
import { Modal } from './Modal';

/** Minimal Term reference — decoupled from any specific page's node/row shape. */
export type SqlAttributesModalTerm = {
	id: string;
	name: string;
};

type SqlAttributesModalProps = {
	term: SqlAttributesModalTerm | null;
	onClose: () => void;
};

// The caller constrains the width; `Text` clips to one line and
// reveals the full text in a popover on hover.
const TruncatedDescription = ({ text }: { text: string | null }) => {
	const value = text?.trim() ?? '';

	if (value === '') {
		return <span className="italic text-secondary dark:text-zinc-500">No Description</span>;
	}

	return <Text text={value} />;
};

/** Generic modal listing a Term's SQL Attributes. Reusable from any page that has a term id. */
export const SqlAttributesModal = ({ term, onClose }: SqlAttributesModalProps) => {
	const termId = term?.id ?? null;

	const fetchPage = useCallback(
		async (skip: number, limit: number) => {
			if (termId == null) return { items: [], total: 0 };
			const response = await termsApi.getSqlAttributes(termId, { skip, limit });
			if (response.error) {
				return { error: response.message ?? 'Failed to load SQL attributes' };
			}
			return { items: response.data ?? [], total: response.total ?? 0 };
		},
		[termId],
	);

	const {
		items: attributes,
		total,
		isLoading: loading,
		isLoadingMore,
		error,
		hasMore,
		loadMore,
	} = useInfiniteList(fetchPage, {
		enabled: termId != null,
		itemKey: (attribute) => attribute.id,
	});

	return (
		<Modal open={term != null} onClose={onClose} className="w-full max-w-2xl">
			<header className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex min-w-0 items-center gap-2">
					<Icon
						name={IconName.Link}
						className="h-5 w-5 shrink-0 text-body dark:text-zinc-300"
					/>
					<Text as="h2" variant={TextVariant.Heading}>
						{term?.name} — SQL Attributes ({total})
					</Text>
				</div>
				<Button
					theme={ButtonTheme.IconNeutral}
					size={Size.SMALL}
					iconOnly
					type="button"
					onClick={onClose}
					aria-label="Close SQL attributes"
				>
					<Icon name={IconName.Close} className="h-4 w-4" />
				</Button>
			</header>
			<InfiniteScroll
				className="max-h-[70dvh] p-5"
				onLoadMore={loadMore}
				isLoading={isLoadingMore}
				hasMore={hasMore}
				// Only a failed *first* page is shown here — a failed later page
				// keeps the list below and gets its own retry control instead (see
				// `error` on `InfiniteScroll`).
				error={attributes.length > 0 ? error : null}
			>
				{loading ? (
					<div role="status" aria-label="Loading SQL attributes">
						<SkeletonSqlBlocks withHeading />
					</div>
				) : error != null && attributes.length === 0 ? (
					<p className="text-sm text-red-600 dark:text-red-300">{error}</p>
				) : attributes.length === 0 ? (
					<EmptyState variant={EmptyStateVariant.Inline} title="No SQL Attributes" />
				) : (
					<ul className="space-y-4">
						{attributes.map((attr) => (
							<li
								key={attr.id}
								className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700"
							>
								<Text as="h3" text={attr.name} variant={TextVariant.Heading} />
								<div className="mt-1 max-w-md text-xs text-secondary dark:text-zinc-400">
									<TruncatedDescription text={attr.description} />
								</div>
								<SqlBlock
									className="mt-2"
									sql={attr.expression || attr.sql || ''}
									label="SQL"
								/>
							</li>
						))}
					</ul>
				)}
			</InfiniteScroll>
		</Modal>
	);
};
