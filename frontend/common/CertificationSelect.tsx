// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState } from 'react';
import { Spinner } from '@nvidia/foundations-react-core';
import { CertificationBadge, CERTIFICATION_LABEL } from '@/common/CertificationBadge';
import { PopoverMenu } from '@/common/PopoverMenu';
import { Icon, IconName } from '@/common/icons';
import { fieldStatus } from '@/lib/certification';

export type CertificationSelectProps = {
	/** Current value of the single boolean flag this control edits. */
	certified: boolean;
	/**
	 * Called with the newly selected value (Pending -> false, Certified -> true).
	 * A returned promise keeps the control in a saving state until it settles,
	 * so each control tracks its own write without the parent holding per-row state.
	 */
	onChange: (certified: boolean) => void | Promise<void>;
	/** When true, renders a read-only badge instead of the dropdown. */
	disabled?: boolean;
	/** Renders the full pill (icon + status text) instead of the icon-only badge. */
	showLabel?: boolean;
};

export const CertificationSelect = ({
	certified,
	onChange,
	disabled = false,
	showLabel = false,
}: CertificationSelectProps) => {
	const [saving, setSaving] = useState(false);
	const status = fieldStatus(certified);

	if (disabled) {
		return <CertificationBadge status={status} iconOnly={!showLabel} />;
	}

	const select = async (next: boolean) => {
		if (saving) return;
		setSaving(true);
		try {
			await onChange(next);
		} finally {
			setSaving(false);
		}
	};

	const items = [
		{
			label: 'Pending Approval',
			icon: <Icon name={IconName.Certification} className="h-4 w-4 text-secondary" />,
			onClick: () => {
				void select(false);
			},
		},
		{
			label: 'Certified',
			icon: <Icon name={IconName.Certification} className="h-4 w-4 text-[#76b900]" />,
			onClick: () => {
				void select(true);
			},
		},
	];

	return (
		<PopoverMenu
			className="inline-flex"
			items={items}
			trigger={({ toggle }) => (
				<button
					type="button"
					onClick={toggle}
					disabled={saving}
					// The badge inside is decorative, so the trigger has to name the
					// status it is about to change as well as the action.
					title={
						saving
							? 'Saving Certification…'
							: `Set Certification (${CERTIFICATION_LABEL[status]})`
					}
					className={`inline-flex items-center gap-1 ${saving ? 'cursor-default opacity-70' : 'cursor-pointer'}`}
				>
					<CertificationBadge status={status} iconOnly={!showLabel} decorative />
					{saving ? (
						<Spinner aria-label="Saving certification" className="h-3.5 w-3.5" />
					) : (
						<Icon
							name={IconName.ChevronRight}
							className="h-3 w-3 rotate-90 text-secondary"
						/>
					)}
				</button>
			)}
		/>
	);
};
