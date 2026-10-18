// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ApiResponse } from './types';
import type { Owner } from '@/types/owners';

export const ownersApi = {
	/** Every user and every group, with the permissions of their role. */
	list: (): Promise<ApiResponse<Owner[]>> => requests.get('owners'),
};
