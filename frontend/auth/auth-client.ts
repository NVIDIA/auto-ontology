// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAuthClient } from 'better-auth/react';
import { adminClient } from 'better-auth/client/plugins';
import { ssoClient } from '@better-auth/sso/client';
import { ac, roles } from '@/auth/auth-access';

export const authClient = createAuthClient({
	plugins: [adminClient({ ac, roles }), ssoClient()],
});

export const { signIn, signOut, useSession } = authClient;
