// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { requireUser } from '@/auth/auth-guards';

export const metadata: Metadata = {
	title: 'Account',
};

// Unlike /settings (admin-only), /account is every signed-in user's own space.
// Minting an API token still requires apiToken:manage, which a viewer lacks.
export default async function AccountLayout({ children }: { children: React.ReactNode }) {
	await requireUser();
	return children;
}
