// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { redirect } from 'next/navigation';
import { connectionsApi } from '@/api/connections';
import { ConnectionsView } from '@/components/connectionsPage/ConnectionsView';

export default async function ConnectionsSettingsPage() {
	// Connections from CONNECTION_STRINGS are fixed by config and cannot be
	// created, edited or removed here, which is why SettingsNav hides this
	// section. Hiding the link is not enough on its own: a bookmark or a typed
	// URL would still reach a page whose every action is a dead end.
	const envManaged = await connectionsApi.isEnvSource();
	if (envManaged === true) {
		redirect('/settings/zones');
	}

	return (
		<main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]">
			<ConnectionsView />
		</main>
	);
}
