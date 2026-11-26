// SPDX-FileCopyrightText: Copyright (c) 2026, AkilsuryaS.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	resolve: { alias: { '@': fileURLToPath(new URL('../', import.meta.url)) } },
	test: { environment: 'node', include: ['tests/model-import.test.ts'] },
});
