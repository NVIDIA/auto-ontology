// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { type ReactNode } from 'react';
import { BackPanelLayout } from '@/common/BackPanelLayout';
import { SettingsNav } from '@/components/settings/SettingsNav';

type SettingsPanelLayoutProps = {
	children: ReactNode;
	/** Passed straight to the nav, which drops Connections when it is set. */
	connectionsEnvManaged: boolean;
};

export const SettingsPanelLayout = ({
	children,
	connectionsEnvManaged,
}: SettingsPanelLayoutProps) => (
	<BackPanelLayout
		panelAriaLabel="Settings sections"
		expandAriaLabel="Expand settings menu"
		collapseAriaLabel="Collapse settings menu"
		panel={<SettingsNav connectionsEnvManaged={connectionsEnvManaged} />}
	>
		{children}
	</BackPanelLayout>
);
