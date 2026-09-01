// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { ExplorationView } from '@/components/explorationPage';
import { ExplorationLoader } from '@/components/explorationPage/ExplorationLoader';

export default function ExplorationPage() {
	return (
		<Suspense fallback={<ExplorationLoader />}>
			<ExplorationView />
		</Suspense>
	);
}
