// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requireAdmin } from '@/auth/auth-guards';
import { getPrisma } from '@/lib/prisma';
import { SsoConfigForm } from './SsoConfigForm';

const AdminSsoPage = async () => {
	await requireAdmin();
	// Fetch on the server so the form renders with the correct state on first
	// paint (no flash of the empty registration form before the client fetch).
	const providers = await getPrisma().ssoProvider.findMany({
		select: { providerId: true, issuer: true, domain: true },
	});
	return <SsoConfigForm initialProviders={providers} />;
};

export default AdminSsoPage;
