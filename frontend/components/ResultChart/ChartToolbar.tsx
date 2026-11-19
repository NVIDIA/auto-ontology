// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { FC } from 'react';
import { ChevronDown, ChevronUp, Download } from '@/adapters/ui/icons';
import { Button } from '@/common/Button';
import { ButtonTheme, Size } from '@/enums/button';
import { downloadCsv } from './export-csv';
import type { ChartSpec } from './types';

/** Per-chart actions: reveal the underlying rows, or export them as CSV. */
export const ChartToolbar: FC<{
	spec: ChartSpec;
	dataOpen: boolean;
	dataId: string;
	onToggleData: () => void;
}> = ({ spec, dataOpen, dataId, onToggleData }) => (
	<div className="result-chart-toolbar">
		<Button
			theme={ButtonTheme.Secondary}
			size={Size.SMALL}
			iconPosition="left"
			aria-expanded={dataOpen}
			aria-controls={dataOpen ? dataId : undefined}
			onClick={onToggleData}
		>
			{dataOpen ? (
				<ChevronUp className="h-3.5 w-3.5" />
			) : (
				<ChevronDown className="h-3.5 w-3.5" />
			)}
			{dataOpen ? 'Hide data' : 'Show data'}
		</Button>
		<div className="ml-auto">
			<Button
				theme={ButtonTheme.Secondary}
				size={Size.SMALL}
				iconPosition="left"
				title="Download the Chart Data as CSV"
				onClick={() => downloadCsv(spec)}
			>
				<Download className="h-3.5 w-3.5" />
				CSV
			</Button>
		</div>
	</div>
);
