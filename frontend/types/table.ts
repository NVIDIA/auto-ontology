// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

export type TableColumn<T> = {
	/** Stable identifier for the column (used as the React key). */
	key: string;
	/** Header cell content. */
	header: ReactNode;
	/** Renders the body cell for a given row. */
	cell: (row: T) => ReactNode;
	/**
	 * Value revealed in the tooltip of a `truncate` column, for cells whose
	 * rendered content differs from the value worth reading. Defaults to the
	 * cell's own text. Ignored on other columns.
	 */
	title?: (row: T) => string;
	/** Tailwind width class for the column, e.g. `w-44`. */
	width?: string;
	/** Clips content to a single line and reveals it in a tooltip on hover, only when clipped. */
	truncate?: boolean;
	/** Width cap for a truncated cell. Defaults to `max-w-0` (table-fixed). */
	maxWidthClass?: string;
	/** Prevent the cell content from wrapping. */
	nowrap?: boolean;
	/** Extra classes applied to the body cell (e.g. text colour). */
	className?: string;
	/** Extra classes applied to the header cell. */
	headerClassName?: string;
};

export type TableRangeMeta = {
	firstItemIndex: number;
	lastItemIndex: number;
	totalItems: number;
};

export type TablePagination = {
	/** 1-based current page. */
	page: number;
	pageSize: number;
	/** Total number of rows across all pages. */
	totalItems: number;
	onPageChange: (page: number) => void;
	/** Overrides the "1-10 out of 62" range label. */
	rangeTextFormatFn?: (meta: TableRangeMeta) => ReactNode;
};

export type TableProps<T> = {
	columns: TableColumn<T>[];
	/** Rows for the current page (already sliced when paginating). */
	rows: T[];
	/** Extracts a stable React key for a row. */
	rowKey: (row: T, index: number) => string;
	/** When provided, renders a paginated footer (the Kaizen pager). */
	pagination?: TablePagination;
	/** Column sizing strategy. Defaults to `fixed` (table-fixed). */
	layout?: 'fixed' | 'auto';
	/** Table text size class. Defaults to `text-sm`. */
	textClassName?: string;
	/** Padding applied to every header and body cell. Defaults to `px-4 py-3`. */
	cellClassName?: string;
	/** Minimum table width, e.g. `min-w-[28rem]`. */
	minWidthClass?: string;
	/** Replaces the default outer card classes when set. */
	containerClassName?: string;
	/** Header row classes (text styling inherits into the cells). */
	theadClassName?: string;
	/** Body (`tbody`) classes. Defaults to a divided list. */
	bodyClassName?: string;
	/** Per-row classes. Defaults to hover styling. */
	rowClassName?: string;
	/** Wraps the table in a scroll container, e.g. `max-h-[478px] overflow-auto`. */
	scrollClassName?: string;
	/** Extra classes appended to the outer container (e.g. margins). */
	className?: string;
	/** When set, rows become clickable and receive pointer/hover affordances. */
	onRowClick?: (row: T, index: number) => void;
	/**
	 * Set when a cell already links to where the row click goes. The row click
	 * stays a pointer shortcut and the row keeps its table semantics: claiming
	 * the keyboard on it would add a tab stop next to the link's, and turn a
	 * row holding a link and a menu into a button with focusable children.
	 */
	rowClickIsPointerShortcut?: boolean;
	/** Empty-state title shown instead of the table when `rows` is empty. Defaults to `—`. */
	emptyMessage?: ReactNode;
};
