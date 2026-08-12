// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ConnectionType } from '@/enums/connection';

export type DatabricksConnectionParams = {
	type: ConnectionType.DATABRICKS;
	host: string;
	http_path: string;
	password: string;
	database: string;
	/** Optional ingestion allowlist: only these schemas are ingested. Empty/absent = all. */
	schemas?: string[];
	/**
	 * Single schema to ingest, entered on the form instead of picking from a list.
	 * Verified by the connection test, then sent as `schemas`; never persisted itself.
	 */
	schema?: string;
	/**
	 * Run chat queries as the signed-in user by exchanging their SSO token for a
	 * Databricks token, instead of using the stored access token. Ingestion is
	 * unaffected and always uses the stored token.
	 */
	sso_federation?: boolean;
};

export type PostgresConnectionParams = {
	type: ConnectionType.POSTGRESQL;
	host: string;
	port: string;
	user: string;
	password: string;
	database: string;
};

export type SnowflakeConnectionParams = {
	type: ConnectionType.SNOWFLAKE;
	account: string;
	warehouse: string;
	user: string;
	password: string;
	database: string;
	/** Optional ingestion allowlist: only these schemas are ingested. Empty/absent = all. */
	schemas?: string[];
};

export type HeavyDBConnectionParams = {
	type: ConnectionType.HEAVYDB;
	host: string;
	port: string;
	user: string;
	password: string;
	database: string;
	protocol: string;
};

/** Structured connection form fields, discriminated by `type`. */
export type ConnectionParams =
	| DatabricksConnectionParams
	| PostgresConnectionParams
	| SnowflakeConnectionParams
	| HeavyDBConnectionParams;

/** A stored connection returned by the API. */
export type Connection = {
	database_name: string;
	connection: ConnectionParams;
};

/** User-provided fields for testing or creating a connection. */
export type ConnectionInput = ConnectionParams;
