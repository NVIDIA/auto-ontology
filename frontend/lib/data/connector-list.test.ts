// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';
import type { Database } from '../../types/datasources';
import { connectorsToCsv, searchConnectors } from './connector-list';

const db = (name: string, extra: Partial<Database> = {}): Database => ({
	id: name,
	name,
	num_of_schemas: 1,
	schemas: [],
	...extra,
});

const dbs = [db('beta'), db('alpha'), db('Gamma')];

describe('searchConnectors', () => {
	it('matches the name ignoring case and surrounding space', () => {
		expect(searchConnectors(dbs, ' AM').map((d) => d.name)).toEqual(['Gamma']);
	});

	it('returns every connector for a blank query', () => {
		expect(searchConnectors(dbs, '  ')).toEqual(dbs);
	});

	it('returns nothing when no name matches', () => {
		expect(searchConnectors(dbs, 'zzz')).toEqual([]);
	});
});

describe('connectorsToCsv', () => {
	it('writes a header and quotes cells', () => {
		const csv = connectorsToCsv([db('a"b', { dialect: 'snowflake' })], []);
		expect(csv.split('\r\n')).toEqual([
			'"Name","Connector","Owner","Created","Last Pulled"',
			'"a""b","Snowflake","Unassigned","",""',
		]);
	});

	it('neutralises spreadsheet formulas', () => {
		expect(connectorsToCsv([db('=SUM(A1)')], [])).toContain(`"'=SUM(A1)"`);
	});
});
