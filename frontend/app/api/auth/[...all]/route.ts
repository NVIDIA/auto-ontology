// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { toNextJsHandler } from 'better-auth/next-js';
import { auth } from '@/auth/auth';

export const { GET, POST } = toNextJsHandler(auth);
