// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { User } from '@/types/auth';

export type ConversationAnalytics = {
	id: string;
	user: User;
	source: string;
	question: string;
	questionTimestamp: string;
	response: string | null;
	responseTimestamp: string | null;
	sql: string | null;
};
