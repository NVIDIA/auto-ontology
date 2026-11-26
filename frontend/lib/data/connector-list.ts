// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Relative imports, and no `@/lib/owners`, keep this module loadable by the
// plain vitest run, which has no `@/` alias.
import { connectionDisplayName, isConnectionType } from '../../enums/connection';
import type { Database } from '../../types/datasources';
import type { Owner } from '../../types/owners';

export const connectorKey = (database: Database): string =>
	isConnectionType(database.dialect) ? database.dialect : '';

export const connectorLabel = (key: string): string =>
	isConnectionType(key) ? connectionDisplayName[key] : 'Other';

/** The connectors whose name contains `query`, ignoring case; all when it is blank. */
export const searchConnectors = (databases: Database[], query: string): Database[] => {
	const needle = query.trim().toLowerCase();
	return needle === ''
		? databases
		: databases.filter((database) => database.name.toLowerCase().includes(needle));
};

// A cell that opens with one of these is run as a formula by spreadsheets.
const FORMULA_START = /^[=+\-@\t\r]/;

const csvCell = (value: string): string => {
	const safe = FORMULA_START.test(value) ? `'${value}` : value;
	return `"${safe.replaceAll('"', '""')}"`;
};

/** The given connectors as CSV text: name, connector, owner, created, last pulled. */
export const connectorsToCsv = (databases: Database[], owners: Owner[]): string => {
	const header = ['Name', 'Connector', 'Owner', 'Created', 'Last Pulled'];
	const rows = databases.map((database) => [
		database.name,
		connectorLabel(connectorKey(database)),
		owners.find((owner) => owner.id === database.owner_id)?.name ?? 'Unassigned',
		database.created ?? '',
		database.pulled ?? '',
	]);
	return [header, ...rows].map((row) => row.map(csvCell).join(',')).join('\r\n');
};

/** Hands the browser `text` as a file to save. */
export const downloadTextFile = (filename: string, text: string, mime: string): void => {
	const url = URL.createObjectURL(new Blob([text], { type: mime }));
	const link = document.createElement('a');
	link.href = url;
	link.download = filename;
	document.body.appendChild(link);
	link.click();
	link.remove();
	URL.revokeObjectURL(url);
};
