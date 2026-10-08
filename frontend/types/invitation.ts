// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Role } from '@/enums/auth';
import type { InvitationStatus } from '@/enums/invitation';

export type Invitation = {
	id: string;
	email: string;
	name: string;
	role: Role;
	status: InvitationStatus;
	expires_at: string;
	created_at: string;
	/** Present only while the invite is active, so the admin can copy the link. */
	url: string | null;
};
