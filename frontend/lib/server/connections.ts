// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Server reads of the connections API that more than one component makes in
 * the same render.
 *
 * Import these from server components only — `cache()` is React's per-request
 * memo, and it has nothing to memoize in a browser.
 */

import { cache } from 'react';

import { connectionsApi } from '@/api/connections';

/**
 * Whether connections come from `CONNECTION_STRINGS` rather than the UI.
 *
 * Four server components ask: the settings layout shapes its nav by it, and
 * the section index, the Connections page and Semantic Compilation each
 * decide what to render or where to redirect. A layout and the page inside it
 * render in one pass, so without this they asked the backend the same
 * question twice per navigation — `cache()` collapses that to once, and every
 * caller in the pass is answered from the same read, which is also what stops
 * two of them from disagreeing.
 */
export const readConnectionsEnvSource = cache(() => connectionsApi.isEnvSource());
