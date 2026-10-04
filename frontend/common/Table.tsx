// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import {
	PaginationArrowButton,
	PaginationItemRangeText,
	PaginationNavigationGroup,
	PaginationPageList,
	PaginationRoot,
} from '@nvidia/foundations-react-core';

import { EmptyState } from '@/common/EmptyState';
import { Text } from '@/common/Text';
import { EmptyStateVariant } from '@/enums/emptyState';
import type { TableColumn, TableProps } from '@/types/table';

const DEFAULT_CONTAINER =
	'overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900';

export const TABLE_THEAD_CLASSNAME =
	'border-b border-zinc-200 bg-zinc-50 text-left text-xs font-semibold text-body dark:border-zinc-800 dark:bg-zinc-800/60 dark:text-zinc-400';

const DEFAULT_TBODY = 'divide-y divide-zinc-100 dark:divide-zinc-800';

const DEFAULT_ROW = 'transition-colors hover:bg-zinc-50 dark:hover:bg-zinc-800/40';

const cx = (...classes: (string | false | undefined)[]) => classes.filter(Boolean).join(' ');

// Rows without a click handler shouldn't get a hover affordance, even when
// `rowClassName` (or the default) bakes in `hover:`/`dark:hover:` utilities.
const stripHoverClasses = (classes: string): string =>
	classes
		.split(/\s+/)
		.filter((token) => token !== '' && !token.includes('hover:'))
		.join(' ');

const bodyCellClasses = <T,>(column: TableColumn<T>, cellClassName: string): string =>
	cx(
		cellClassName,
		'align-top',
		column.nowrap && 'whitespace-nowrap',
		column.truncate && (column.maxWidthClass ?? 'max-w-0'),
		column.className,
	);

const defaultRangeText = ({
	firstItemIndex,
	lastItemIndex,
	totalItems,
}: {
	firstItemIndex: number;
	lastItemIndex: number;
	totalItems: number;
}) => `${firstItemIndex}-${lastItemIndex} out of ${totalItems}`;

export const Table = <T,>({
	columns,
	rows,
	rowKey,
	pagination,
	layout = 'fixed',
	textClassName = 'text-sm',
	cellClassName = 'px-4 py-3',
	minWidthClass,
	containerClassName,
	theadClassName,
	bodyClassName,
	rowClassName,
	scrollClassName,
	className,
	onRowClick,
	rowClickIsPointerShortcut = false,
	emptyMessage,
}: TableProps<T>) => {
	const baseRowClassName = rowClassName ?? DEFAULT_ROW;
	const interactiveRowClassName = onRowClick
		? cx(baseRowClassName, 'cursor-pointer')
		: stripHoverClasses(baseRowClassName);
	// A row answers the keyboard only when it is the sole way into its target.
	const rowIsButton = onRowClick != null && !rowClickIsPointerShortcut;

	if (rows.length === 0) {
		return (
			<div className={cx(containerClassName ?? DEFAULT_CONTAINER, className)}>
				<EmptyState variant={EmptyStateVariant.Inline} title={emptyMessage ?? '—'} />
			</div>
		);
	}

	const tableElement = (
		<table
			className={cx(
				'w-full',
				layout === 'fixed' && 'table-fixed',
				minWidthClass,
				textClassName,
			)}
		>
			<thead>
				<tr className={theadClassName ?? TABLE_THEAD_CLASSNAME}>
					{columns.map((column) => (
						<th
							key={column.key}
							className={cx(cellClassName, column.width, column.headerClassName)}
						>
							{column.header}
						</th>
					))}
				</tr>
			</thead>
			<tbody className={bodyClassName ?? DEFAULT_TBODY}>
				{rows.map((row, index) => (
					<tr
						key={rowKey(row, index)}
						className={interactiveRowClassName}
						onClick={
							onRowClick
								? (e) => {
										// A modified click asks for another tab or
										// window, which a scripted navigation cannot
										// give — so it is left to a link in the row
										// rather than answered in this one.
										if (e.metaKey || e.ctrlKey || e.shiftKey) return;
										onRowClick(row, index);
									}
								: undefined
						}
						onKeyDown={
							rowIsButton
								? (e) => {
										// Only the row's own keystrokes. Enter and Space
										// belong to whatever is focused, so a nested
										// control — an action menu, a link in a cell —
										// would otherwise have its activation swallowed
										// by `preventDefault` and navigate the row
										// instead.
										if (e.target !== e.currentTarget) return;
										if (e.key === 'Enter' || e.key === ' ') {
											e.preventDefault();
											onRowClick(row, index);
										}
									}
								: undefined
						}
						role={rowIsButton ? 'button' : undefined}
						tabIndex={rowIsButton ? 0 : undefined}
					>
						{columns.map((column) => (
							<td key={column.key} className={bodyCellClasses(column, cellClassName)}>
								{column.truncate ? (
									<Text text={column.title?.(row)} focusable>
										{column.cell(row)}
									</Text>
								) : (
									column.cell(row)
								)}
							</td>
						))}
					</tr>
				))}
			</tbody>
		</table>
	);

	return (
		<div className={cx(containerClassName ?? DEFAULT_CONTAINER, className)}>
			{scrollClassName ? <div className={scrollClassName}>{tableElement}</div> : tableElement}
			{pagination && (
				<PaginationRoot
					totalItems={pagination.totalItems}
					pageSize={pagination.pageSize}
					page={pagination.page}
					onPageChange={pagination.onPageChange}
					className="border-t border-zinc-200 px-4 py-3 dark:border-zinc-800"
				>
					<PaginationNavigationGroup withTabs>
						<PaginationArrowButton direction="previous" />
						<PaginationPageList />
						<PaginationArrowButton direction="next" />
					</PaginationNavigationGroup>
					<PaginationItemRangeText
						rangeTextFormatFn={pagination.rangeTextFormatFn ?? defaultRangeText}
					/>
				</PaginationRoot>
			)}
		</div>
	);
};
