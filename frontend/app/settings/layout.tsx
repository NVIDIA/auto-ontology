// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { connectionsApi } from '@/api/connections';
import { SettingsPanelLayout } from '@/components/settings/SettingsPanelLayout';
import { requireAdmin } from '@/auth/auth-guards';

export const metadata: Metadata = {
	title: 'Settings',
};

// The entire Settings section (Connections, Zones, Users, SSO) is admin-only.
// This is the real gate; the nav rail also hides Settings from non-admins.
export default async function SettingsLayout({ children }: { children: React.ReactNode }) {
	await requireAdmin();
	// Read once here so the nav renders its final shape on the first paint, and
	// so navigating between sections does not re-ask: the layout survives those
	// navigations, the nav's own effect did not.
	const envManaged = await connectionsApi.isEnvSource();
	return (
		<SettingsPanelLayout connectionsEnvManaged={envManaged === true}>
			{children}
		</SettingsPanelLayout>
	);
}
