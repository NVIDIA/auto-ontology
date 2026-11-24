// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';
import { parseSqlResponse } from '../../lib/parseSqlResponse';
import { specToCsv, tableToCsv } from './export-csv';
import type { ChartSpec } from './types';

describe('tableToCsv', () => {
	it('keeps column order and names the file from the title', () => {
		const { filename, csv } = tableToCsv(
			{ columns: ['b', 'a'], rows: [{ a: '1', b: '2' }] },
			'Query Results 20261119-1155',
		);
		expect(filename).toBe('query-results-20261119-1155.csv');
		expect(csv).toBe('b,a\r\n2,1');
	});

	it('quotes commas, quotes and newlines', () => {
		const { csv } = tableToCsv(
			{ columns: ['v'], rows: [{ v: 'a,b' }, { v: 'say "hi"' }, { v: 'x\ny' }] },
			't',
		);
		expect(csv).toBe('v\r\n"a,b"\r\n"say ""hi"""\r\n"x\ny"');
	});

	it('neutralizes formulas but keeps real numbers', () => {
		const { csv } = tableToCsv(
			{
				columns: ['v'],
				rows: [
					{ v: '=SUM(A1)' },
					{ v: '+1+1' },
					{ v: '@cmd' },
					{ v: '-x' },
					{ v: '-5' },
					{ v: '\n=1' },
				],
			},
			't',
		);
		expect(csv).toBe("v\r\n'=SUM(A1)\r\n'+1+1\r\n'@cmd\r\n'-x\r\n-5\r\n\"'\n=1\"");
	});

	it('keeps zero-padded codes as text', () => {
		const { csv } = tableToCsv({ columns: ['zip'], rows: [{ zip: '007' }] }, 't');
		expect(csv).toBe('zip\r\n007');
	});

	it('serializes a __proto__ column as its row value', () => {
		const rows = JSON.parse('[{"__proto__":"x","a":"1"}]');
		const { csv } = tableToCsv({ columns: ['__proto__', 'a'], rows }, 't');
		expect(csv).toBe('__proto__,a\r\nx,1');
	});

	it('keeps a __proto__ column from a parsed SQL response', () => {
		const table = parseSqlResponse('[{"__proto__":"x","a":"1"}]');
		expect(table).not.toBeNull();
		expect(tableToCsv(table!, 't').csv).toBe('__proto__,a\r\nx,1');
	});

	it('falls back to query-results when the title has no visible characters', () => {
		expect(tableToCsv({ columns: ['a'], rows: [] }, '!!!').filename).toBe('query-results.csv');
	});
});

describe('specToCsv', () => {
	it('serializes x then series with labels as headers', () => {
		const spec: ChartSpec = {
			type: 'bar',
			title: 'Revenue by Region',
			x: { key: 'region', label: 'Region' },
			series: [{ key: 'rev', label: 'Revenue' }, { key: 'cost' }],
			data: [
				{ region: 'EMEA', rev: 10, cost: -2 },
				{ region: '=bad', rev: null, cost: 3 },
			],
		};
		expect(specToCsv(spec)).toEqual({
			filename: 'revenue-by-region.csv',
			csv: "Region,Revenue,cost\r\nEMEA,10,-2\r\n'=bad,,3",
		});
	});

	it('still falls back to chart for an untitled spec', () => {
		const spec: ChartSpec = { type: 'bar', title: '', x: { key: 'x' }, series: [], data: [] };
		expect(specToCsv(spec).filename).toBe('chart.csv');
	});
});
