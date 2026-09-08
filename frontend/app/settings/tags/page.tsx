// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { TagsSettingsSkeleton, TagsSettingsView } from './TagsSettingsView';

/**
 * The view reads the open tag from `?focus=`, and `useSearchParams` has to sit
 * under a Suspense boundary — without one the search params are not available
 * to prerender and the build fails on this route. Same arrangement as
 * `app/terms/page.tsx`, which selects a term the same way.
 */
export default function TagsSettingsPage() {
	return (
		<Suspense fallback={<TagsSettingsSkeleton />}>
			<TagsSettingsView />
		</Suspense>
	);
}
