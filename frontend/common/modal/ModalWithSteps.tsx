// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, type ReactNode } from 'react';
import { Icon, IconName } from '@/common/icons';
import { Modal } from './Modal';

export type StepperFooterAction = {
	label: string;
	onClick: () => void;
	disabled?: boolean;
	loading?: boolean;
	variant?: 'primary' | 'outline';
};

export type ModalWithStepsProps = {
	open: boolean;
	onClose: () => void;
	title: string;
	titleIcon?: IconName;
	steps: string[];
	activeStep: number;
	onActiveStepChange?: (step: number) => void;
	disabledSteps?: number[];
	children: ReactNode;
	footerActions: StepperFooterAction[];
	alert?: string | null;
	className?: string;
};

enum StepStatus {
	DISABLED = 'disabled',
	ACTIVE = 'active',
	COMPLETED = 'completed',
	INACTIVE = 'inactive',
}

const stepStatusClass: Record<StepStatus, string> = {
	[StepStatus.DISABLED]: 'pointer-events-none text-zinc-400 dark:text-zinc-600',
	[StepStatus.ACTIVE]: 'pointer-events-none font-medium text-zinc-900 dark:text-zinc-100',
	[StepStatus.COMPLETED]:
		'cursor-pointer text-zinc-600 hover:rounded-lg hover:bg-zinc-100 hover:text-[#76b900] dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-[#8fd100]',
	[StepStatus.INACTIVE]: 'pointer-events-none text-zinc-400 dark:text-zinc-500',
};

const getStepStatus = (index: number, activeStep: number, disabledSteps: number[]): StepStatus => {
	if (disabledSteps.includes(index)) return StepStatus.DISABLED;
	if (activeStep === index) return StepStatus.ACTIVE;
	if (activeStep > index) return StepStatus.COMPLETED;
	return StepStatus.INACTIVE;
};

export const ModalWithSteps = ({
	open,
	onClose,
	title,
	titleIcon = IconName.Database,
	steps,
	activeStep,
	onActiveStepChange,
	disabledSteps = [],
	children,
	footerActions,
	alert = null,
	className = 'flex w-[650px] max-w-full flex-col',
}: ModalWithStepsProps) => {
	const contentRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		contentRef.current?.scrollTo(0, 0);
	}, [activeStep]);

	return (
		<Modal open={open} onClose={onClose} className={className}>
			<div className="flex items-center justify-between border-b border-zinc-200 px-5 py-4 dark:border-zinc-700">
				<div className="flex items-center gap-2">
					<Icon name={titleIcon} className="h-5 w-5 text-[#76b900]" />
					<h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{title}
					</h3>
				</div>
				<button
					type="button"
					onClick={onClose}
					className="cursor-pointer rounded-md p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-zinc-300"
					aria-label="Close"
				>
					<svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
						<path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
					</svg>
				</button>
			</div>

			<nav
				className="flex min-h-[42px] flex-wrap items-center gap-1 px-5 py-2"
				aria-label="Connection wizard steps"
			>
				{steps.map((label, index) => {
					const status = getStepStatus(index, activeStep, disabledSteps);
					return (
						<div key={label} className="flex items-center gap-1">
							<button
								type="button"
								disabled={status !== StepStatus.COMPLETED}
								onClick={() => {
									if (status === StepStatus.COMPLETED) {
										onActiveStepChange?.(index);
									}
								}}
								className={`px-2 py-1 text-sm capitalize ${stepStatusClass[status]}`}
							>
								{label}
							</button>
							{index < steps.length - 1 && (
								<Icon
									name={IconName.ChevronRight}
									className="h-3.5 w-3.5 shrink-0 text-zinc-300 dark:text-zinc-600"
									aria-hidden
								/>
							)}
						</div>
					);
				})}
			</nav>

			<div className="border-t border-zinc-200 dark:border-zinc-700" />

			<div
				ref={contentRef}
				className="flex min-h-[405px] max-h-[min(70vh,520px)] flex-col overflow-y-auto p-3"
			>
				{children}
			</div>

			<div className="border-t border-zinc-200 dark:border-zinc-700" />

			<div className="flex min-h-[60px] items-center gap-3 px-4 py-3">
				{alert != null && (
					<p className="min-w-0 flex-1 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
						{alert}
					</p>
				)}
				<div className={`flex flex-1 justify-end gap-2 ${alert == null ? '' : 'shrink-0'}`}>
					{footerActions.map((action) => {
						const isPrimary = action.variant !== 'outline';
						const disabled = action.disabled || action.loading;
						return (
							<button
								key={action.label}
								type="button"
								data-testid={`stepper-button-${action.label}`}
								onClick={action.onClick}
								disabled={disabled}
								className={
									isPrimary
										? `cursor-pointer rounded-lg px-4 py-2 text-sm font-medium transition-colors ${disabled ? 'cursor-default bg-zinc-200 text-zinc-500 dark:bg-zinc-700 dark:text-zinc-400' : 'bg-[#76b900] text-white hover:bg-[#5e9400]'}`
										: `cursor-pointer rounded-lg border border-[#76b900] bg-white px-4 py-2 text-sm font-medium text-[#5e9400] transition-colors hover:bg-[#76b900]/10 disabled:cursor-default disabled:border-zinc-300 disabled:text-zinc-400 dark:border-[#76b900] dark:bg-zinc-900 dark:hover:bg-[#76b900]/20 disabled:dark:border-zinc-600`
								}
							>
								{action.loading ? `${action.label}…` : action.label}
							</button>
						);
					})}
				</div>
			</div>
		</Modal>
	);
};
