// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { Icon, IconName } from '@/common/icons';
import { ExplorationLayer } from '@/enums/exploration';

type ViewToggleProps = {
	layer: ExplorationLayer;
	onToggle: () => void;
};

export const ViewToggle = ({ layer, onToggle }: ViewToggleProps) => {
	const isSemantic = layer === ExplorationLayer.Semantic;

	return (
		<button
			type="button"
			onClick={onToggle}
			className="flex h-10 cursor-pointer items-center gap-2 whitespace-nowrap rounded-lg border border-[#76b900] bg-[#76b900] px-3 text-sm font-medium text-white shadow-md transition-colors hover:bg-[#5e9400]"
		>
			<Icon name={isSemantic ? IconName.Database : IconName.Terms} className="h-4 w-4" />
			Switch to {isSemantic ? 'Data Objects' : 'Semantic Objects'}
		</button>
	);
};
