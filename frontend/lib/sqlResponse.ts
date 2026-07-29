// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The agent returns the executed-DB rows under `sql_response_from_db`. It can
// be a stringified markdown/CSV table or a structured ``list[dict]`` payload.
// Normalise both shapes into a single string so DB persistence and parsing in
// `DynamicTable` stay simple (compact JSON for objects → cheap to re-parse).
//
// Shared between `useChat.ts` (client-side render of the result event) and
// the chat-completions proxy route (server-side persistence of the same
// event), so both sides store/display the identical format.
export const stringifySqlResponse = (value: unknown): string | undefined => {
	if (value == null) return undefined;
	if (typeof value === 'string') return value.trim() ? value : undefined;
	try {
		return JSON.stringify(value);
	} catch {
		return undefined;
	}
};
