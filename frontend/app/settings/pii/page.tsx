// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requireAdmin } from '@/auth/auth-guards';
import { isPiiDetectionEnabled } from '@/lib/configurations';
import { PiiSettingsForm } from './PiiSettingsForm';

const PiiSettingsPage = async () => {
	await requireAdmin();
	// Fetch on the server so the toggle renders in the correct state on first
	// paint, with no client-side flash. Absent reads as off — see the route for
	// why this flag is opt-in.
	return <PiiSettingsForm initialEnabled={await isPiiDetectionEnabled()} />;
};

export default PiiSettingsPage;
