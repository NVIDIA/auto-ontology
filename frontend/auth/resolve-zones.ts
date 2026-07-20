// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Zone membership is no longer an authorization boundary. Both roles receive
 * the unfiltered catalog scope.
 */
export function resolveZoneIds(userId: string, role: string | null): string[] | null {
	void userId;
	void role;
	return null;
}
