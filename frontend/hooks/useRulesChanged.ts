// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

/**
 * Telling whatever lists rules that a rule was written.
 *
 * A rule is created from the global search, and the search opens from the top
 * bar on every screen — including the Rules settings screen, which offers no
 * create button of its own once it has rows to show. So the component that
 * saves a rule is generally not the one listing them, sits outside it in the
 * tree, and cannot be handed a callback: hence a subscription held by the
 * module rather than passed down.
 *
 * At most one listener is live, since the list is a single mount, and none at
 * all while no Rules screen is open — which makes an announcement free.
 *
 * Deliberately not a cache: it carries no rule and does not say what changed.
 * The list re-reads its first page, which is the only thing that can tell where
 * a new rule sorts and how many there now are.
 */

import { useEffect, useRef } from 'react';

const listeners = new Set<() => void>();

/** Announce a rule written, for whatever is listing them to re-read. */
export const notifyRulesChanged = (): void => {
	listeners.forEach((listener) => listener());
};

/** Run *onChange* whenever a rule is written, from anywhere in the app. */
export const useRulesChanged = (onChange: () => void): void => {
	// Subscribed once and read through a ref, so a caller passing the handler
	// inline — the natural way to write it — doesn't resubscribe every render.
	const handler = useRef(onChange);

	useEffect(() => {
		handler.current = onChange;
	}, [onChange]);

	useEffect(() => {
		const listener = () => handler.current();
		listeners.add(listener);
		return () => {
			listeners.delete(listener);
		};
	}, []);
};
