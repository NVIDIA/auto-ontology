// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metadata } from 'next';
import { ApiTokensView } from '@/components/apiTokensPage/ApiTokensView';

export const metadata: Metadata = {
	title: 'API Tokens',
};

export default function ApiTokensPage() {
	return <ApiTokensView />;
}
