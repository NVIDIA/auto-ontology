// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type Prompt = {
	id: string;
	content: string;
};

export type Acronym = {
	id: string;
	name: string;
	description: string;
	createdAt: string;
	updatedAt: string;
};

async function json<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
	const res = await fetch(input, init);
	if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
	return res.json() as Promise<T>;
}

export const promptsApi = {
	get: () => json<Prompt[]>('/api/custom-prompts'),

	create: (data: { content?: string } = {}) =>
		json<Prompt>('/api/custom-prompts', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		}),

	update: (id: string, data: { content?: string }) =>
		json<Prompt>(`/api/custom-prompts/${id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		}),

	delete: (id: string) => fetch(`/api/custom-prompts/${id}`, { method: 'DELETE' }),
};

export const acronymsApi = {
	get: () => json<Acronym[]>('/api/acronyms'),

	checkName: (name: string) => json<{ exists: boolean }>(`/api/acronyms?name=${name}`),

	create: (data: { name: string; description?: string }) =>
		json<Acronym>('/api/acronyms', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		}),

	update: (id: string, data: { name?: string; description?: string }) =>
		json<Acronym>(`/api/acronyms/${id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		}),

	delete: (id: string) => fetch(`/api/acronyms/${id}`, { method: 'DELETE' }),
};

export const semanticCompilationApi = {
	get: () => json<{ enabled: boolean }>('/api/configurations/semantic-compilation'),

	getStatus: () => json<{ calculated: boolean }>('/api/semantic-compilation/status'),

	setEnabled: (enabled: boolean) =>
		json<{ enabled: boolean }>('/api/configurations/semantic-compilation', {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ enabled }),
		}),
};
