// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { oauthProviderAuthServerMetadata } from '@better-auth/oauth-provider';
import { auth } from '@/auth/auth';

// RFC 8414 path-aware discovery for Better Auth's issuer, APP_URL/api/auth: a
// client inserts `/.well-known/oauth-authorization-server` between the issuer's
// origin and its path, and looks here first. Better Auth answers this path
// itself, but in Next.js the request never reaches its `api/auth/[...all]`
// handler, so without this route the client gets a 404. The MCP server
// advertises this issuer as its authorization server (#560).
export const GET = oauthProviderAuthServerMetadata(auth);
