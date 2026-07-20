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

import type { TableColumn, TableProps } from '@/types/table';

const DEFAULT_CONTAINER =
	'overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900';

const DEFAULT_THEAD =
	'border-b border-zinc-200 bg-zinc-50 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:border-zinc-800 dark:bg-zinc-800/60 dark:text-zinc-400';

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
	emptyMessage,
}: TableProps<T>) => {
	const baseRowClassName = rowClassName ?? DEFAULT_ROW;
	const interactiveRowClassName = onRowClick
		? cx(baseRowClassName, 'cursor-pointer')
		: stripHoverClasses(baseRowClassName);

	if (rows.length === 0) {
		return (
			<div className={cx(containerClassName ?? DEFAULT_CONTAINER, className)}>
				<p className="p-4 text-sm italic text-zinc-500 dark:text-zinc-400">
					{emptyMessage ?? '—'}
				</p>
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
				<tr className={theadClassName ?? DEFAULT_THEAD}>
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
						onClick={onRowClick ? () => onRowClick(row, index) : undefined}
						onKeyDown={
							onRowClick
								? (e) => {
										if (e.key === 'Enter' || e.key === ' ') {
											e.preventDefault();
											onRowClick(row, index);
										}
									}
								: undefined
						}
						role={onRowClick ? 'button' : undefined}
						tabIndex={onRowClick ? 0 : undefined}
					>
						{columns.map((column) => (
							<td key={column.key} className={bodyCellClasses(column, cellClassName)}>
								{column.truncate ? (
									<span
										className="block truncate"
										title={column.title?.(row) ?? ''}
									>
										{column.cell(row)}
									</span>
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
