// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { redirect } from 'next/navigation';
import { connectionsApi } from '@/api/connections';

// Settings is admin-only (enforced by layout.tsx). Land on the first section
// the nav actually offers: SettingsNav drops Connections when they come from
// CONNECTION_STRINGS, so a fixed redirect there opened a page with no entry in
// the menu beside it. Both read the same flag so they cannot disagree.
export default async function SettingsPage() {
	const envManaged = await connectionsApi.isEnvSource();
	redirect(envManaged === true ? '/settings/zones' : '/settings/connections');
}
