// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { SettingsPanelLayout } from '@/components/settings/SettingsPanelLayout';
import { requireAdmin } from '@/auth/auth-guards';
import { readConnectionsEnvSource } from '@/lib/server/connections';

export const metadata: Metadata = {
	title: 'Settings',
};

// The entire Settings section (Connections, Zones, Users, SSO) is admin-only.
// This is the real gate; the nav rail also hides Settings from non-admins.
export default async function SettingsLayout({ children }: { children: React.ReactNode }) {
	await requireAdmin();
	// Read once here so the nav renders its final shape on the first paint, and
	// so navigating between sections does not re-ask: the layout survives those
	// navigations, the nav's own effect did not. The page rendering inside this
	// pass asks the same question, which `readConnectionsEnvSource` answers
	// from this read rather than a second one.
	//
	// Only the flag. The connections *list* is read by the root layout, which
	// puts it in the query cache for the Connections page to subscribe to —
	// this one is a plain server value the nav reads while rendering, so it
	// stays a direct await.
	const envManaged = await readConnectionsEnvSource();

	return (
		<SettingsPanelLayout connectionsEnvManaged={envManaged === true}>
			{children}
		</SettingsPanelLayout>
	);
}
