// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { connectionsApi } from '@/api/connections';
import { requireAdmin } from '@/auth/auth-guards';
import { getPrisma } from '@/lib/prisma';
import { SemanticCompilationForm } from './SemanticCompilationForm';

const CONFIG_KEY = 'semantic_compilation_enabled';

const SemanticCompilationPage = async () => {
	await requireAdmin();
	// Fetch on the server so the toggle (and the connected-database hint)
	// render in the correct state on first paint, with no client-side flash.
	const [row, connections, isEnvSource] = await Promise.all([
		getPrisma().configuration.findUnique({ where: { key: CONFIG_KEY } }),
		connectionsApi.getAll(),
		connectionsApi.isEnvSource(),
	]);
	// `connections.count` only reflects UI-managed connections (stored on the
	// catalog database row / Vault) — it's always 0 when connections instead
	// come from CONNECTION_STRINGS, since that path never writes either. Treat
	// env-managed mode as "databases connected" too, otherwise this would tell
	// an operator to "connect a data source" they already configured.
	const hasDatabases = (!connections.error && connections.count > 0) || isEnvSource === true;
	return (
		<SemanticCompilationForm
			initialEnabled={row?.value === 'true'}
			hasDatabases={hasDatabases}
		/>
	);
};

export default SemanticCompilationPage;
