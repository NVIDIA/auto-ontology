// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Connector kinds supported in the new-connection wizard. */
export enum ConnectionType {
	DATABRICKS = 'databricks',
	POSTGRESQL = 'postgresql',
	SNOWFLAKE = 'snowflake',
	HEAVYDB = 'heavydb',
}

export const connectionDisplayName: Record<ConnectionType, string> = {
	[ConnectionType.DATABRICKS]: 'Databricks',
	[ConnectionType.POSTGRESQL]: 'PostgreSQL',
	[ConnectionType.SNOWFLAKE]: 'Snowflake',
	[ConnectionType.HEAVYDB]: 'HeavyDB',
};

export const isConnectionType = (value: string | null | undefined): value is ConnectionType =>
	typeof value === 'string' && (Object.values(ConnectionType) as string[]).includes(value);

export type ConnectionFieldKey =
	| 'host'
	| 'http_path'
	| 'port'
	| 'account'
	| 'warehouse'
	| 'user'
	| 'password'
	| 'database'
	| 'protocol';

export type ConnectionField = {
	key: ConnectionFieldKey;
	label: string;
	placeholder?: string;
	secret?: boolean;
	/** Optional fields are not required to enable Test/Create. */
	optional?: boolean;
};

/** Form fields rendered per connector type. `database` is the connection identity. */
export const CONNECTION_FIELDS: Record<ConnectionType, ConnectionField[]> = {
	[ConnectionType.DATABRICKS]: [
		{
			key: 'host',
			label: 'Server hostname',
			placeholder: 'dbc-a1b2345c-d6e7.cloud.databricks.com',
		},
		{
			key: 'http_path',
			label: 'HTTP path',
			placeholder: '/sql/1.0/warehouses/a1b234c567d8e9fa',
		},
		{ key: 'password', label: 'Access token', secret: true },
		{ key: 'database', label: 'Catalog', placeholder: 'main' },
	],
	[ConnectionType.POSTGRESQL]: [
		{ key: 'host', label: 'Host', placeholder: 'localhost' },
		{ key: 'port', label: 'Port', placeholder: '5432' },
		{ key: 'user', label: 'User', placeholder: 'postgres' },
		{ key: 'password', label: 'Password', secret: true },
		{ key: 'database', label: 'Database', placeholder: 'my_database' },
	],
	[ConnectionType.SNOWFLAKE]: [
		{ key: 'account', label: 'Account', placeholder: 'xy12345.us-east-1' },
		{ key: 'warehouse', label: 'Warehouse', placeholder: 'COMPUTE_WH' },
		{ key: 'user', label: 'User' },
		{ key: 'password', label: 'Password', secret: true },
		{ key: 'database', label: 'Database', placeholder: 'MY_DATABASE' },
	],
	[ConnectionType.HEAVYDB]: [
		{ key: 'host', label: 'Host', placeholder: 'localhost' },
		{ key: 'port', label: 'Port', placeholder: '6274', optional: true },
		{ key: 'user', label: 'User', placeholder: 'admin' },
		{ key: 'password', label: 'Password', secret: true },
		{ key: 'database', label: 'Database', placeholder: 'heavyai' },
		{ key: 'protocol', label: 'Protocol', placeholder: 'binary', optional: true },
	],
};
