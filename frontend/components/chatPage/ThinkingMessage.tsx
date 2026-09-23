// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { GraphStep } from '@/types/chat';
import { Icon, IconName } from '@/common/icons';
import { SqlBlock } from '@/common/SqlBlock';

type ThinkingMessageProps = {
	steps: GraphStep[];
	/**
	 * The validated query the agent is running, shown while it executes.
	 * Replaced in place if execution fails and the query is rebuilt, so a run
	 * that retries doesn't stack up discarded attempts.
	 */
	liveSql?: string | null;
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

export const ThinkingMessage = ({ steps, liveSql }: ThinkingMessageProps) => {
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
						Auto Ontology Agent
					</span>
				</div>

				{completedSteps.length > 0 && (
					<ul className="mb-2 space-y-1">
						{completedSteps.map((step, i) => (
							<li
								key={`${step.node}-${i}`}
								className="flex items-start gap-2 text-xs text-zinc-500 dark:text-zinc-400"
							>
								<Icon
									name={IconName.Check}
									className="mt-0.5 h-3 w-3 shrink-0 text-[#76b900]"
								/>
								<span>
									<span>{step.label}</span>
									{step.thought && (
										<span className="block text-zinc-400 dark:text-zinc-500">
											{step.thought}
										</span>
									)}
								</span>
							</li>
						))}
					</ul>
				)}

				<div className="flex w-full flex-col items-start gap-2">
					{activeStep && (
						<span className="text-sm text-zinc-800 dark:text-zinc-100">
							<span>{activeStep.label}</span>
							{activeStep.thought && (
								<span className="block text-xs font-normal text-zinc-500 dark:text-zinc-400">
									{activeStep.thought}
								</span>
							)}
						</span>
					)}

					{/* Below the active step, not above it: the query is cleared
					    for execution by the step currently showing, so rendering
					    it higher up would read as having happened first. */}
					{liveSql && (
						<SqlBlock
							sql={liveSql}
							label={
								liveSql.trim().toUpperCase().startsWith('PREDICT') ? 'PQL' : 'SQL'
							}
							className="w-full"
						/>
					)}

					<span className="flex items-end gap-1">
						<Dot delay="0s" />
						<Dot delay="0.2s" />
						<Dot delay="0.4s" />
					</span>
				</div>
			</div>
		</div>
	);
};
