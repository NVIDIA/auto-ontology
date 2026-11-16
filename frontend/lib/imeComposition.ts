// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { KeyboardEvent } from 'react';

/**
 * True while an input method (Korean, Japanese, Chinese, …) is composing text.
 *
 * Key handlers must ignore these events: the Enter that confirms a composed
 * syllable is otherwise read as submit or commit, sending the message early and
 * leaving the last syllable behind. `keyCode === 229` covers Safari, which
 * delivers the confirming keydown after `compositionend`, when `isComposing` is
 * already false.
 */
export function isImeComposing(e: KeyboardEvent): boolean {
	return e.nativeEvent.isComposing || e.keyCode === 229;
}
