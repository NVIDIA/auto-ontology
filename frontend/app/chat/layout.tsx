// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { requireUser } from '@/auth/auth-guards';

export const metadata: Metadata = {
	title: 'Chat',
};

export default async function ChatLayout({ children }: { children: React.ReactNode }) {
	// Redirect to /login when there's no valid session (e.g. an expired or
	// rotated-secret cookie) instead of rendering the chat shell unauthenticated.
	await requireUser();
	return children;
}
