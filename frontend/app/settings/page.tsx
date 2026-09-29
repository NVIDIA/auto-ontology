// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { redirect } from 'next/navigation';
import { readConnectionsEnvSource } from '@/lib/server/connections';

// Settings is admin-only (enforced by layout.tsx). Land on the first section
// the nav actually offers: SettingsNav drops Connections when they come from
// CONNECTION_STRINGS, so a fixed redirect there opened a page with no entry in
// the menu beside it. Both read the same flag so they cannot disagree — now
// literally, since the layout rendering around this reads it through the same
// per-request memo.
export default async function SettingsPage() {
	const envManaged = await readConnectionsEnvSource();
	redirect(envManaged === true ? '/settings/zones' : '/settings/connections');
}
