// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Suspense, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { Icon, IconName } from '@/common/icons';
import { deliverLoopbackCallback, isLoopbackCallback } from '@/auth/oauth-loopback';
import { OauthStatus } from '@/app/oauth/OauthStatus';

// The page is public and takes its target from the query string, so it only
// ever talks to a loopback listener. Anything else is refused rather than
// navigated to; the server-side rewrite never sends other URLs here.
const callbackError = (target: string | null): string | null => {
	if (!target) return 'Missing callback.';
	try {
		if (!isLoopbackCallback(new URL(target))) return 'Invalid callback.';
	} catch {
		return 'Invalid callback.';
	}
	return null;
};

const HandoffStatus = () => {
	const params = useSearchParams();
	const target = params.get('url');
	const error = callbackError(target);

	useEffect(() => {
		if (error || !target) return undefined;
		void deliverLoopbackCallback(target);
		return undefined;
	}, [error, target]);

	if (error) {
		return <p className="text-center text-sm text-red-500">{error}</p>;
	}

	return (
		<OauthStatus title="Thank You">
			Authentication successful. You can close this page.
		</OauthStatus>
	);
};

const HandoffPage = () => (
	<div className="flex h-full w-full items-center justify-center bg-white p-6 dark:bg-zinc-950">
		<div className="flex flex-col items-center gap-6">
			<div className="flex items-center gap-2">
				<Icon name={IconName.NvidiaLogo} className="h-6 w-6" />
				<span className="text-lg font-semibold text-heading dark:text-zinc-100">
					Auto Ontology
				</span>
			</div>
			<Suspense fallback={null}>
				<HandoffStatus />
			</Suspense>
		</div>
	</div>
);

export default HandoffPage;
