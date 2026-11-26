// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { oauthProviderAuthServerMetadata } from '@better-auth/oauth-provider';
import { auth } from '@/auth/auth';

// Publishes Auto Ontology's OAuth authorization-server metadata (RFC 8414) so an MCP
// client can discover how to sign a user in.
//
// The issuer is APP_URL/api/auth, and its path-aware location is served by
// `api/auth/route.ts` beside this file. This root copy is for clients that look
// for the document at the origin root, as older ones do: without it they find
// the page auth gate instead and are redirected to /login, so discovery fails
// before it starts. `/.well-known` is exempted from that gate in proxy.ts; the
// document is public by design and contains no secrets.
export const GET = oauthProviderAuthServerMetadata(auth);
