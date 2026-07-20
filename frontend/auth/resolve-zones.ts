// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Server-side helper that resolves the zone IDs accessible to the current user.
//
// Admins have unrestricted access to all data — returns null (no filter).
// Viewers receive only the zones they have been explicitly granted access to;
// returns the list of zone IDs that should be forwarded to data endpoints so
// the Python backend can scope results without doing a per-request user lookup.
//
// This follows the same pattern as illumex's NestJS BFF: the BFF resolves zone
// IDs from user membership once per request and forwards them as a filter param,
// keeping data endpoints free of user-identity concerns.

const PYTHON_API_URL = process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001';

type ZoneRow = { id: string };
type ZonesResponse = { data: ZoneRow[] };

async function fetchUserZoneIds(userId: string): Promise<string[]> {
	try {
		const url = `${PYTHON_API_URL}/api/zones?uid=${encodeURIComponent(userId)}`;
		const res = await fetch(url, { headers: { Accept: 'application/json' } });
		if (!res.ok) {
			throw new Error(`Failed to resolve accessible zones: ${res.status}`);
		}
		const json = (await res.json()) as ZonesResponse;
		return (json.data ?? []).map((z) => z.id);
	} catch (error) {
		throw new Error('Failed to resolve accessible zones', { cause: error });
	}
}

/**
 * Return the zone IDs the given user can access, or `null` for admins (no filter).
 *
 * Calls the internal Python `/api/zones?uid=<userId>` endpoint which already
 * implements the admin-sees-all / viewer-sees-granted logic.
 */
export async function resolveZoneIds(
	userId: string,
	role: string | null,
): Promise<string[] | null> {
	if (role === 'admin') return null;
	return fetchUserZoneIds(userId);
}
