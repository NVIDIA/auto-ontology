// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { SettingsPanelLayout } from '@/components/settings/SettingsPanelLayout';
import { requireAdmin } from '@/auth/auth-guards';

export const metadata: Metadata = {
	title: 'Settings',
};

// The entire Settings section (Connections, Zones, Users, SSO) is admin-only.
// This is the real gate; the nav rail also hides Settings from non-admins.
export default async function SettingsLayout({ children }: { children: React.ReactNode }) {
	await requireAdmin();
	return <SettingsPanelLayout>{children}</SettingsPanelLayout>;
}
