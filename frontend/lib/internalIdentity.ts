// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const INTERNAL_USER_HEADER = 'x-auto-ontology-user-id';

/**
 * Forward the authenticated user identity across the private Next.js →
 * FastAPI hop. FastAPI must not be publicly reachable because this header is
 * trusted without independent verification.
 */
export const buildInternalIdentityHeaders = (userId: string): Record<string, string> => ({
	[INTERNAL_USER_HEADER]: userId,
});
