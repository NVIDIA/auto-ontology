// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Typed client for the GSF query helpers (AI SQL translation). */

import { requests } from './requests';

export type ExplainResponse = {
	explanation: string;
};

/**
 * Translate a single SQL statement into a plain-English summary via the
 * backend NIM LLM. Errors are returned on the response object (`error: true`)
 * by the shared requests wrapper rather than thrown.
 */
export const explainQuery = (sql: string, abortController?: AbortController) =>
	requests.post<ExplainResponse>('/queries/explain', { sql }, abortController);
