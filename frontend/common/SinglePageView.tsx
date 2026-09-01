// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useRef, useState } from 'react';
import {
	SinglePageComposer,
	type ComposerEditValue,
	type SinglePageComposerProps,
} from '@/common/SinglePageComposer';
import { SkeletonDetail } from '@/common/Skeleton';

export type SinglePageFormat = SinglePageComposerProps;

export type SinglePageViewProps = {
	dataId: string;
	getSinglePage: (dataId: string, treeFocusId: string | null) => Promise<SinglePageFormat>;
	treeFocusId?: string | null;
	/** Increment when explorer tree merges API data so details re-render without changing focus. */
	treeDataEpoch?: number;
	isEditing?: boolean;
	onPatchEdits?: (
		edits: Record<string, ComposerEditValue>,
	) => Promise<{ error?: boolean; message?: string }>;
	onSave?: (edits: Record<string, ComposerEditValue>) => void;
	onCancel?: () => void;
	onDataTableRowClick?: (sectionId: string, rowId: string) => void;
	onEditSql?: (sectionId: string, sql: string) => void;
	onSuggestDescription?: (sectionId: string) => Promise<string | null>;
	onCertificationChange?: (id: string, certified: boolean) => void | Promise<void>;
	onDataTableCertificationChange?: (
		sectionId: string,
		rowId: string,
		certified: boolean,
	) => void | Promise<void>;
	inlineSaveSectionId?: string;
	hideEditToolbar?: boolean;
};

export const SinglePageView = ({
	dataId,
	getSinglePage,
	treeFocusId = null,
	treeDataEpoch = 0,
	isEditing = false,
	onPatchEdits,
	onSave,
	onCancel,
	onDataTableRowClick,
	onEditSql,
	onSuggestDescription,
	onCertificationChange,
	onDataTableCertificationChange,
	inlineSaveSectionId,
	hideEditToolbar,
}: SinglePageViewProps): React.JSX.Element | null => {
	const [loading, setLoading] = useState(true);
	const [props, setProps] = useState<SinglePageFormat | null>(null);
	const prevCoreRef = useRef<{
		dataId: string;
		treeFocusId: string | null;
		getSinglePage: (dataId: string, treeFocusId: string | null) => Promise<SinglePageFormat>;
	} | null>(null);

	useEffect(() => {
		let cancelled = false;
		const prev = prevCoreRef.current;
		const coreChanged =
			prev == null ||
			prev.dataId !== dataId ||
			prev.treeFocusId !== treeFocusId ||
			prev.getSinglePage !== getSinglePage;
		prevCoreRef.current = { dataId, treeFocusId, getSinglePage };

		(async () => {
			if (coreChanged) {
				setLoading(true);
			}
			try {
				const response = await getSinglePage(dataId, treeFocusId);
				if (!cancelled) {
					setProps(response);
				}
			} finally {
				if (!cancelled) {
					setLoading(false);
				}
			}
		})();
		return () => {
			cancelled = true;
		};
	}, [dataId, getSinglePage, treeFocusId, treeDataEpoch]);

	if (loading) {
		return (
			<div className="flex flex-1" role="status" aria-label="Loading details">
				<SkeletonDetail />
			</div>
		);
	}

	if (!props) {
		return null;
	}

	const hasTreeFocus = treeFocusId != null && treeFocusId !== '';

	return (
		<div className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
			<SinglePageComposer
				header={{
					header: {
						...(hasTreeFocus ? { entityId: dataId } : {}),
						...props.header.header,
					},
					errorBanner: props.header.errorBanner,
				}}
				sections={props.sections}
				rightPanel={props?.rightPanel}
				leftPanel={props?.leftPanel}
				entityUpdatingProperties={props?.entityUpdatingProperties}
				isEditingMode={isEditing}
				onPatchEdits={onPatchEdits}
				onSave={onSave}
				onCancel={onCancel}
				onDataTableRowClick={onDataTableRowClick}
				onEditSql={onEditSql}
				onSuggestDescription={onSuggestDescription}
				onCertificationChange={onCertificationChange}
				onDataTableCertificationChange={onDataTableCertificationChange}
				inlineSaveSectionId={inlineSaveSectionId}
				hideEditToolbar={hideEditToolbar}
			/>
		</div>
	);
};
