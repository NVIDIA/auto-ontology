// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { EmptyState } from '@/common/EmptyState';
import { Icon, IconName } from '@/common/icons';
import { EmptyStateVariant } from '@/enums/emptyState';

// Blocks the chat conversation area when the semantic layer hasn't been
// calculated yet: the agent can't answer questions without it.
export const SemanticNotReady = () => (
	<EmptyState
		variant={EmptyStateVariant.Welcome}
		illustration={
			<div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#76b900]/15">
				<Icon name={IconName.NvidiaLogo} className="h-6 w-6 text-[#76b900]" />
			</div>
		}
		title="Semantic layer not ready"
		description="The semantic layer hasn't been created yet, so I can't answer questions."
	/>
);
