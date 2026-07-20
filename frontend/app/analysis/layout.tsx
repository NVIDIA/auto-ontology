// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { requireUser } from '@/auth/auth-guards';

export const metadata: Metadata = {
	title: 'Analysis',
};

export default async function AnalysisLayout({ children }: { children: React.ReactNode }) {
	await requireUser();
	return children;
}
