// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const TICK_MS = 100;
const COMPLETE_HOLD_MS = 280;
const DEFAULT_CAP = 90;
const DEFAULT_TAU_MS = 3500;

export type StartSmoothProgressOptions = {
	/**
	 * Time constant in milliseconds. Larger values climb toward `cap` more
	 * slowly. Tune this to the expected duration of the request.
	 */
	tauMs?: number;
};

/**
 * Drive a determinate progress bar while a request has no byte-level percent.
 *
 * The value eases toward `cap` (default 90) for as long as the work runs, then
 * `complete()` snaps to 100 so the user sees a finished bar before it unmounts.
 */
export const useSmoothProgress = (cap = DEFAULT_CAP) => {
	const [value, setValue] = useState(0);
	const activeRef = useRef(false);
	const startedAtRef = useRef(0);
	const tauMsRef = useRef(DEFAULT_TAU_MS);
	const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

	const clearTick = useCallback(() => {
		if (intervalRef.current === null) return;
		clearInterval(intervalRef.current);
		intervalRef.current = null;
	}, []);

	const start = useCallback(
		({ tauMs = DEFAULT_TAU_MS }: StartSmoothProgressOptions = {}) => {
			clearTick();
			activeRef.current = true;
			tauMsRef.current = tauMs;
			startedAtRef.current = Date.now();
			setValue(1);
			intervalRef.current = setInterval(() => {
				if (!activeRef.current) return;
				const elapsed = Date.now() - startedAtRef.current;
				const next = cap * (1 - Math.exp(-elapsed / tauMsRef.current));
				setValue(Math.max(1, Math.min(cap, next)));
			}, TICK_MS);
		},
		[cap, clearTick],
	);

	const fail = useCallback(() => {
		activeRef.current = false;
		clearTick();
		setValue(0);
	}, [clearTick]);

	const complete = useCallback(async () => {
		activeRef.current = false;
		clearTick();
		setValue(100);
		await new Promise<void>((resolve) => {
			setTimeout(resolve, COMPLETE_HOLD_MS);
		});
	}, [clearTick]);

	useEffect(() => () => clearTick(), [clearTick]);

	return { value, start, complete, fail };
};
