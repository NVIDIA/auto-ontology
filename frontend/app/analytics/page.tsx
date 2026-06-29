// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requireAdmin } from '@/auth/auth-guards';
import { AnalyticsView } from '@/components/analyticsPage';

// Analytics is admin-only; non-admins are redirected to /chat.
export default async function AnalyticsPage() {
	await requireAdmin();
	return <AnalyticsView />;
}
