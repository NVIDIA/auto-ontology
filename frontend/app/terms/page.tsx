// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { redirect } from 'next/navigation';

import { termHref } from '@/lib/termHref';

const first = (value: string | string[] | undefined): string | undefined =>
	Array.isArray(value) ? value[0] : value;

/**
 * Terms now open on the Exploration page, beside their node in the graph.
 * Old links — bookmarks, tag pages, shared URLs — keep working: `focus`
 * becomes `semanticId`, and `sqlAttr`/`colAttr` carry over unchanged.
 */
export default async function TermsPage({
	searchParams,
}: {
	searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
	const params = await searchParams;
	const focus = first(params.focus);
	if (!focus) redirect('/exploration');
	redirect(termHref(focus, { sqlAttr: first(params.sqlAttr), colAttr: first(params.colAttr) }));
}
