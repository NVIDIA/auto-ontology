// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useMemo, type FC } from 'react';
import { DynamicTable } from '@/components/chatPage/DynamicTable';
import type { ParsedTable } from '@/lib/parseSqlResponse';
import type { ChartSpec } from './types';

/** The chart's exact underlying rows, shown by the "Show data" toggle. */
export const ChartDataTable: FC<{ spec: ChartSpec; id?: string }> = ({ spec, id }) => {
	const table = useMemo<ParsedTable>(() => {
		const columns = [spec.x.key, ...spec.series.map((s) => s.key)];
		const rawHeaders = [
			spec.x.label ?? spec.x.key,
			...spec.series.map((s) => s.label ?? s.key),
		];

		// DynamicTable keys each row by its header text, so two series that
		// happen to share a display label would otherwise collide into one
		// column. Disambiguate repeats deterministically before building rows.
		const seenCounts = new Map<string, number>();
		const headers = rawHeaders.map((header) => {
			const count = seenCounts.get(header) ?? 0;
			seenCounts.set(header, count + 1);
			return count === 0 ? header : `${header} (${count + 1})`;
		});

		const headerByKey = Object.fromEntries(columns.map((key, i) => [key, headers[i]]));

		return {
			columns: headers,
			rows: spec.data.map((row) => {
				const out: Record<string, string> = {};
				for (const key of columns) {
					const header = headerByKey[key] ?? key;
					out[header] = row[key] == null ? '' : String(row[key]);
				}
				return out;
			}),
		};
	}, [spec]);

	return (
		<div className="mt-3" role="region" aria-label="Chart data" id={id}>
			<div className="mb-1 text-xs font-medium text-zinc-500 dark:text-zinc-400">
				Query results
			</div>
			<DynamicTable table={table} />
		</div>
	);
};
