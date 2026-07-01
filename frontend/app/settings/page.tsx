// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { redirect } from 'next/navigation';

// Settings is admin-only (enforced by layout.tsx). Land on the first section.
export default function SettingsPage() {
	redirect('/settings/connections');
}
