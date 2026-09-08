// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// tagsApi.attach — label an object with an existing tag.
//
// `catalog: ['edit']` rather than `tag: ['manage']`, which is what guards the
// tag list itself: curating the vocabulary is an admin's job, but applying a
// tag that already exists is an edit of the object, no different from editing
// its description. Viewers hold that permission and edit terms with it.
export const POST = withPermission({ catalog: ['edit'] })((req) => proxyToBackend(req));
