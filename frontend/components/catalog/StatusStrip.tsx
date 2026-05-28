// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';

type Status = 'checking' | 'ok' | 'down';

export const StatusStrip = () => {
	const [status, setStatus] = useState<Status>('checking');
	const [version, setVersion] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const r = await fetch('/api/openmetadata/system/version', { cache: 'no-store' });
				if (!r.ok) throw new Error(`status ${r.status}`);
				const json = (await r.json()) as { version?: string };
				if (cancelled) return;
				setVersion(json.version ?? null);
				setStatus('ok');
			} catch {
				if (!cancelled) setStatus('down');
			}
		})();
		return () => {
			cancelled = true;
		};
	}, []);

	const color =
		status === 'ok'
			? 'bg-emerald-500'
			: status === 'down'
				? 'bg-red-500'
				: 'bg-amber-500 animate-pulse';
	const text =
		status === 'ok'
			? `OpenMetadata online${version ? ` · v${version}` : ''}`
			: status === 'down'
				? 'OpenMetadata unreachable — is docker compose up?'
				: 'Connecting…';

	return (
		<div className="ml-auto flex items-center gap-2 px-4 py-2 text-xs text-zinc-500 dark:text-zinc-400">
			<span className={`h-1.5 w-1.5 rounded-full ${color}`} aria-hidden />
			<span>{text}</span>
		</div>
	);
};
