// SPDX-FileCopyrightText: Copyright (c) 2026, AkilsuryaS.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Invoked by the PostgreSQL acceptance harness with a live FastAPI server and
// generated fixtures. Only session resolution is stubbed; permission checks,
// route handlers, proxying, backend validation, and database writes are real.
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/auth/resolve-user', () => ({ resolveUser: vi.fn() }));

import { resolveUser } from '@/auth/resolve-user';
import { POST } from '@/app/api/model/import/route';

const conflictYaml = process.env.MODEL_IMPORT_CONFLICT_YAML;
const boundYaml = process.env.MODEL_IMPORT_BOUND_YAML;
const fixtureState = process.env.MODEL_IMPORT_FIXTURE_STATE;

const expectFixtureUnchanged = async () => {
	const state = await fetch(`${process.env.PYTHON_API_URL}/__fixture_state`);
	expect(state.status).toBe(200);
	expect(await state.json()).toEqual(JSON.parse(fixtureState!));
};

beforeEach(() => {
	vi.mocked(resolveUser).mockResolvedValue({ id: 'acceptance-admin', role: 'admin' });
});

const request = (body: BodyInit) =>
	new Request('http://gateway.test/api/model/import?embed=false&replace=true', {
		method: 'POST',
		body,
	});

describe.skipIf(!conflictYaml || !boundYaml || !fixtureState)(
	'model import gateway with live PostgreSQL backend',
	() => {
		it('preserves the complete typed conflict response', async () => {
			const response = await POST(request(conflictYaml!));
			expect(response.status).toBe(409);
			expect(response.headers.get('cache-control')).toBe('private, no-store');
			const body = await response.json();
			expect(body.detail.code).toBe('model_import_conflict');
			expect(body.detail.conflicts).toHaveLength(7);
			expect(
				body.detail.conflicts.every(
					(c: { existing_ids: string[] }) => c.existing_ids.length > 0,
				),
			).toBe(true);
			expect(body.detail.resolution).toContain('/api/model/export');
			await expectFixtureUnchanged();
		});

		it('preserves conflicts from multipart file uploads', async () => {
			const body = new FormData();
			body.set('file', new Blob([conflictYaml!], { type: 'application/yaml' }), 'model.yaml');
			const response = await POST(request(body));
			expect(response.status).toBe(409);
			expect((await response.json()).detail.conflicts).toHaveLength(7);
			await expectFixtureUnchanged();
		});

		it('imports successfully after binding the exported target IDs', async () => {
			const response = await POST(request(boundYaml!));
			expect(response.status).toBe(200);
			const body = await response.json();
			expect(body.success).toBe(true);
			expect(Object.values(body.summary.created)).toEqual(Array(8).fill(0));
		});

		it('requires the import permission', async () => {
			vi.mocked(resolveUser).mockResolvedValue({ id: 'acceptance-viewer', role: 'viewer' });
			expect((await POST(request(boundYaml!))).status).toBe(403);
		});

		it('requires authentication', async () => {
			vi.mocked(resolveUser).mockResolvedValue(null);
			expect((await POST(request(boundYaml!))).status).toBe(401);
		});
	},
);
