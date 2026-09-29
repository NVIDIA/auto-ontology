// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Next.js instrumentation hook — runs once when the server process starts
 * (not during `next build`). We use it to seed/refresh the bootstrap admin
 * account from AUTO_ONTOLOGY_ADMIN_EMAIL / AUTO_ONTOLOGY_ADMIN_PASSWORD.
 */
export async function register() {
	// Only run in the Node.js server runtime (skip the edge runtime / build).
	if (process.env.NEXT_RUNTIME !== 'nodejs') return;

	try {
		const { seedAdmin } = await import('@/auth/seed-admin');
		await seedAdmin();
	} catch (error) {
		// Don't crash the server if seeding fails (e.g. DB not yet reachable);
		// it will be retried on the next start.
		console.error('[seed-admin] failed to seed bootstrap admin:', error);
	}
}
