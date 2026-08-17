// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Connector kinds supported in the new-connection wizard. */
export enum ConnectionType {
	DATABRICKS = 'databricks',
	POSTGRESQL = 'postgresql',
	SNOWFLAKE = 'snowflake',
	HEAVYDB = 'heavydb',
	SQLITE = 'sqlite',
}

export const connectionDisplayName: Record<ConnectionType, string> = {
	[ConnectionType.DATABRICKS]: 'Databricks',
	[ConnectionType.POSTGRESQL]: 'PostgreSQL',
	[ConnectionType.SNOWFLAKE]: 'Snowflake',
	[ConnectionType.HEAVYDB]: 'HeavyDB',
	[ConnectionType.SQLITE]: 'SQLite',
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
	| 'private_key'
	| 'private_key_passphrase'
	| 'database'
	| 'path'
	| 'protocol'
	| 'schema'
	| 'sso_federation';

export type ConnectionField = {
	key: ConnectionFieldKey;
	label: string;
	placeholder?: string;
	secret?: boolean;
	/** Optional fields are not required to enable Test/Create. */
	optional?: boolean;
	/** Sent with the connection test only; stripped before the connection is created. */
	testOnly?: boolean;
	/** Rendered as a checkbox and sent as a boolean rather than a string. */
	boolean?: boolean;
	/** Rendered as a textarea. Needed for pasted PEM keys, which span many lines. */
	multiline?: boolean;
	/** Helper text shown under the field. */
	hint?: string;
};

/**
 * Fields where supplying any one satisfies the requirement, so none of them can
 * be marked required on its own.
 *
 * Snowflake accounts that enforce MFA reject password sign-in for person users
 * and forbid passwords on service users, leaving a key pair as the only usable
 * credential. Accounts without that enforcement still take a password, so the
 * form has to accept either.
 */
export const CONNECTION_EITHER_FIELDS: Partial<Record<ConnectionType, ConnectionFieldKey[]>> = {
	[ConnectionType.SNOWFLAKE]: ['password', 'private_key'],
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
		{
			key: 'schema',
			label: 'Schema',
			placeholder: 'Leave empty to choose from a list',
			hint: 'Ingest only this schema. The connection test verifies it exists, and the schema selection step is skipped. Leave empty to pick schemas from a list instead.',
			optional: true,
			testOnly: true,
		},
		{
			key: 'sso_federation',
			label: 'Authenticate as signed-in user (SSO)',
			hint: 'Chat queries run with the signed-in user’s own Databricks privileges instead of the access token above. Requires a Databricks federation policy trusting your SSO issuer. Ingestion always uses the access token.',
			optional: true,
			boolean: true,
		},
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
		{
			key: 'password',
			label: 'Password',
			secret: true,
			optional: true,
			hint: 'Leave empty and paste a private key below if the account enforces MFA, which blocks password sign-in for unattended services.',
		},
		{
			key: 'private_key',
			label: 'Private key (PEM)',
			placeholder: '-----BEGIN PRIVATE KEY-----',
			secret: true,
			optional: true,
			multiline: true,
			hint: 'Key-pair authentication. Paste the full PEM for a key registered on the Snowflake user. Used instead of a password.',
		},
		{
			key: 'private_key_passphrase',
			label: 'Private key passphrase',
			secret: true,
			optional: true,
			hint: 'Only needed if the private key above is encrypted.',
		},
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
	[ConnectionType.SQLITE]: [
		{
			key: 'path',
			label: 'File path',
			placeholder: '/data/analytics.sqlite',
			hint: 'Absolute path to the .sqlite file, as seen from the GSF backend and ingestion containers — not from your machine.',
		},
		{
			key: 'database',
			label: 'Database name',
			placeholder: 'analytics',
			hint: 'Names the connection in the catalog. Use the file name without its extension, which is what the connector reports.',
		},
	],
};
