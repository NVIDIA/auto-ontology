// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { oauthProviderAuthServerMetadata } from '@better-auth/oauth-provider';
import { auth } from '@/auth/auth';

// Publishes GSF's OAuth authorization-server metadata (RFC 8414) so an MCP
// client can discover how to sign a user in.
//
// Better Auth already serves this under /api/auth, but a client only knows the
// issuer — which is GSF's origin — and looks for the document at the origin
// root. Without this route the client finds the page auth gate instead and is
// redirected to /login, so discovery fails before it starts. `/.well-known` is
// exempted from that gate in proxy.ts; the document is public by design and
// contains no secrets.
export const GET = oauthProviderAuthServerMetadata(auth);
