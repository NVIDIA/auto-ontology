// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requireAdmin } from '@/auth/auth-guards';
import { UsersManager } from './UsersManager';

const AdminUsersPage = async () => {
	await requireAdmin();
	return <UsersManager />;
};

export default AdminUsersPage;
