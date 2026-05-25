// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { GraphStep } from '@/types/chat';
import { Icon, IconName } from '@/components/icons';

type ThinkingMessageProps = {
	steps: GraphStep[];
};

const Dot = ({ delay }: { delay: string }) => (
	<span
		className="inline-block h-1.5 w-1.5 rounded-full bg-[#76b900]"
		style={{
			animation: 'thinking-bounce 1.2s linear infinite',
			animationDelay: delay,
		}}
	/>
);

export const ThinkingMessage = ({ steps }: ThinkingMessageProps) => {
	const activeStep = [...steps].reverse().find((s) => s.status === 'active');
	const completedSteps = steps.filter((s) => s.status === 'completed');

	return (
		<div className="flex justify-start">
			<style>
				{`@keyframes thinking-bounce {
					0%, 100% { transform: translateY(0); }
					50% { transform: translateY(-5px); }
				}`}
			</style>

			<div className="w-full max-w-[80%] rounded-2xl bg-zinc-100 px-4 py-3 dark:bg-zinc-800">
				<div className="mb-2 flex items-center gap-2">
					<div className="flex h-6 w-6 items-center justify-center rounded-md bg-[#76b900]/15">
						<Icon name={IconName.NvidiaLogo} className="h-4 w-4 text-[#76b900]" />
					</div>
					<span className="text-xs font-semibold text-zinc-700 dark:text-zinc-200">
						GSF Agent
					</span>
				</div>

				{completedSteps.length > 0 && (
					<ul className="mb-2 space-y-1">
						{completedSteps.map((step, i) => (
							<li
								key={`${step.node}-${i}`}
								className="flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400"
							>
								<Icon
									name={IconName.Check}
									className="h-3 w-3 shrink-0 text-[#76b900]"
								/>
								<span>{step.label}</span>
							</li>
						))}
					</ul>
				)}

				<div className="flex items-center gap-2">
					{activeStep && (
						<span className="text-sm text-zinc-800 dark:text-zinc-100">
							{activeStep.label}
						</span>
					)}
					<span className="flex items-end gap-1 pb-0.5">
						<Dot delay="0s" />
						<Dot delay="0.2s" />
						<Dot delay="0.4s" />
					</span>
				</div>
			</div>
		</div>
	);
};
