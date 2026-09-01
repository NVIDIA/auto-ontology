// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { requireUser } from '@/auth/auth-guards';

export const metadata: Metadata = {
	title: 'Account',
};

// Unlike /settings (admin-only), /account is every user's own space — API
// tokens are self-service for viewers too. Any authenticated user passes.
export default async function AccountLayout({ children }: { children: React.ReactNode }) {
	await requireUser();
	return children;
}
