// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { forwardRef, useEffect, useRef, useState, useCallback, type ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { Spinner } from '@nvidia/foundations-react-core';
import type { Breadcrumb } from '@/types/breadcrumbs';
import { ComposerSectionKind } from '@/enums/datasources';
import {
	isComposerSection,
	type ComposerSection,
	type ComposerZonesSection,
} from '@/types/composer-section';
import { Icon, IconName } from '@/common/icons';
import { TagInput } from '@/common/TagInput';
import { Table } from '@/common/Table';
import { TruncatedText } from '@/common/TruncatedText';
import { SqlBlock } from '@/common/SqlBlock';
import { catalogPathFromFocusId } from '@/lib/data/data-catalog-path';
import { datasources } from '@/api/datasources';
import type { NodePatch } from '@/api/types';
import type { TermZone } from '@/types/terms';
import { Toast } from '@/common/Toast';
import { Label } from '@/common/Label';

export type ComposerEditValue = string | string[];

export const LabelList = ({ values }: { values: string[] }) => {
	const nonEmptyValues = values.filter((v) => v.trim() !== '');
	if (nonEmptyValues.length === 0) return <span>—</span>;
	return (
		<ul className="flex flex-wrap gap-1">
			{nonEmptyValues.map((v, i) => (
				<li key={`${v}-${i}`}>
					<Label label={v} maxWidthClass="max-w-[16rem]" />
				</li>
			))}
		</ul>
	);
};

export const ZonesRow = ({ zones }: { zones: TermZone[] }) => (
	<div className="mt-2 flex flex-wrap items-center gap-1.5">
		<span className="text-xs text-zinc-400">Zones:</span>
		{zones.length > 0 ? (
			zones.map((zone) => (
				<Label key={zone.id} label={zone.name} color={zone.color} muted={!zone.enabled} />
			))
		) : (
			<span className="text-xs text-zinc-500 dark:text-zinc-400">-</span>
		)}
	</div>
);

export type SinglePageComposerProps = {
	sections: unknown[];
	header?: {
		header?: Record<string, unknown>;
		errorBanner?: unknown;
	};
	leftPanel?: { bulks: unknown[]; width: string; slot?: ReactNode };
	rightPanel?: { bulks: unknown[]; width: string; slot?: ReactNode };
	pdfProps?: {
		isPDFView: boolean;
		headerProps: Record<string, unknown>;
	};
	entityUpdatingProperties?: Record<string, string | string[]>;
	isEditingMode?: boolean;
	/** Saves pending field edits. Overrides the default `datasources.updateNode` path used for catalog nodes. */
	onPatchEdits?: (
		edits: Record<string, ComposerEditValue>,
	) => Promise<{ error?: boolean; message?: string }>;
	/** Called after Save succeeds, or when Save has nothing to persist. */
	onSave?: (edits: Record<string, ComposerEditValue>) => void;
	onCancel?: () => void;
	onDataTableRowClick?: (sectionId: string, rowId: string) => void;
	onEditSql?: (sectionId: string, sql: string) => void;
	/** Returns an AI-suggested body for a `suggestable` text-card section, or null when none is available. */
	onSuggestDescription?: (sectionId: string) => Promise<string | null>;
	inlineSaveSectionId?: string;
	hideEditToolbar?: boolean;
};

function composerSectionHeading(section: ComposerSection): string {
	switch (section.type) {
		case ComposerSectionKind.LOADING_PANEL:
			return section.message;
		case ComposerSectionKind.ZONES_CHIPS:
			return `${section.title} (${section.zones.length})`;
		case ComposerSectionKind.RELATED_TERMS_CHIPS:
			return `${section.title} (${section.terms.length})`;
		default:
			return section.title;
	}
}

const DescriptionSuggestion = ({
	onSuggest,
	onApply,
}: {
	onSuggest: () => Promise<string | null>;
	onApply: (suggestion: string) => void;
}) => {
	const [suggestion, setSuggestion] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const onSuggestRef = useRef(onSuggest);

	useEffect(() => {
		onSuggestRef.current = onSuggest;
	}, [onSuggest]);

	// Fetch exactly once per mount — this component only exists while the
	// description card is in edit mode, so mounting *is* "entering edit mode".
	// `loading`/`error` already start at their correct values above, so no
	// setState is needed before the async call resolves.
	useEffect(() => {
		let cancelled = false;
		onSuggestRef
			.current()
			.then((result) => {
				if (cancelled) return;
				if (result == null || result.trim() === '') {
					setError('No suggestion available');
					return;
				}
				setSuggestion(result);
			})
			.catch(() => {
				if (!cancelled) setError('Failed to generate a suggestion');
			})
			.finally(() => {
				if (!cancelled) setLoading(false);
			});
		return () => {
			cancelled = true;
		};
	}, []);

	if (!loading && suggestion == null && error == null) return null;

	return (
		<div className="mt-3 rounded-lg border border-[#76b900]/40 bg-[#76b900]/[0.06] p-4 dark:border-[#76b900]/30 dark:bg-[#76b900]/10">
			<div className="flex items-center justify-between gap-3">
				<h3 className="text-xs font-semibold uppercase tracking-wide text-[#4d7a00] dark:text-[#a3d63a]">
					Description Suggestion
				</h3>
				{suggestion != null && (
					<button
						type="button"
						onClick={() => onApply(suggestion)}
						className="shrink-0 cursor-pointer rounded-md border border-[#76b900]/60 bg-white px-2.5 py-1 text-xs font-medium text-[#4d7a00] transition-colors hover:bg-[#76b900]/10 dark:border-[#76b900]/50 dark:bg-zinc-950 dark:text-[#a3d63a] dark:hover:bg-[#76b900]/15"
					>
						Apply as Description
					</button>
				)}
			</div>
			<div className="mt-2 text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
				{loading ? (
					<span className="flex items-center gap-2 italic text-[#4d7a00]/70 dark:text-[#a3d63a]/70">
						<Spinner aria-label="Generating suggestion" className="h-3.5 w-3.5" />
						Generating suggestion…
					</span>
				) : suggestion != null ? (
					suggestion
				) : (
					<span className="italic text-[#4d7a00]/70 dark:text-[#a3d63a]/70">{error}</span>
				)}
			</div>
		</div>
	);
};

const EditableTextCard = ({
	section,
	onChange,
	onSave,
	saving = false,
	autoFocus = false,
	onSuggest,
}: {
	section: { id: string; title: string; body: string; suggestable?: boolean };
	onChange: (sectionId: string, value: string) => void;
	onSave?: () => void;
	saving?: boolean;
	autoFocus?: boolean;
	onSuggest?: () => Promise<string | null>;
}) => {
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const [value, setValue] = useState(section.body);

	useEffect(() => {
		if (autoFocus) textareaRef.current?.focus();
	}, [autoFocus]);

	const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
		setValue(e.target.value);
		onChange(section.id, e.target.value);
	};

	const handleApplySuggestion = (suggestion: string) => {
		setValue(suggestion);
		onChange(section.id, suggestion);
	};

	return (
		<div className="rounded-lg border border-[#76b900]/60 bg-white/90 p-5 shadow-sm ring-1 ring-[#76b900]/10 dark:bg-zinc-950/50">
			<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
				{section.title}
			</h2>
			<textarea
				ref={textareaRef}
				value={value}
				onChange={handleChange}
				rows={4}
				className="mt-3 w-full resize-y rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm leading-relaxed text-zinc-700 outline-none transition-colors focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300"
			/>
			{section.suggestable === true && onSuggest != null && (
				<DescriptionSuggestion onSuggest={onSuggest} onApply={handleApplySuggestion} />
			)}
			{onSave != null && (
				<div className="mt-3 flex justify-end">
					<button
						type="button"
						disabled={saving}
						onClick={onSave}
						className="flex items-center gap-1.5 rounded-lg bg-[#76b900] px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-[#6aa500] disabled:cursor-not-allowed disabled:opacity-70"
					>
						{saving ? (
							<>
								<Spinner aria-label="Saving" className="h-3.5 w-3.5" />
								Saving…
							</>
						) : (
							'Save'
						)}
					</button>
				</div>
			)}
		</div>
	);
};

const EditableTagListCard = ({
	section,
	onChange,
	autoFocus = false,
}: {
	section: { id: string; title: string; values: string[]; hint?: string };
	onChange: (sectionId: string, value: string[]) => void;
	autoFocus?: boolean;
}) => {
	const [tags, setTags] = useState<string[]>(section.values);

	const handleChange = (next: string[]) => {
		setTags(next);
		onChange(section.id, next);
	};

	return (
		<div className="rounded-lg border border-[#76b900]/60 bg-white/90 p-5 shadow-sm ring-1 ring-[#76b900]/10 dark:bg-zinc-950/50">
			<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
				{section.title}
			</h2>
			<div className="mt-3">
				<TagInput
					value={tags}
					onChange={handleChange}
					autoFocus={autoFocus}
					placeholder="Type a value and press Enter"
					ariaLabel={section.title}
				/>
			</div>
			<p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
				{section.hint ??
					'Press Enter or comma to add. Backspace to remove the last tag. Paste comma-separated values to add many at once.'}
			</p>
		</div>
	);
};

const ReadOnlyTagList = ({
	title,
	values,
	sectionId,
}: {
	title: string;
	values: string[];
	sectionId: string;
}) => {
	const nonEmptyValues = values.filter((v) => v.trim() !== '');
	return (
		<div
			id={sectionId === 'sample_values' ? 'sample-values-section' : undefined}
			className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
		>
			<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</h2>
			{nonEmptyValues.length === 0 ? (
				<p className="mt-3 text-sm italic text-zinc-500 dark:text-zinc-400">—</p>
			) : (
				<ul className="mt-3 flex flex-wrap gap-1.5">
					{nonEmptyValues.map((v, i) => (
						<li key={`${v}-${i}`}>
							<Label label={v} maxWidthClass="max-w-[24rem]" />
						</li>
					))}
				</ul>
			)}
		</div>
	);
};

const ZonesSection = ({ section }: { section: ComposerZonesSection }) => {
	const displayedZones = section.zones;

	return (
		<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
			<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
				{section.title} ({displayedZones.length})
			</h2>
			{displayedZones.length === 0 ? (
				<p className="mt-3 text-sm italic text-zinc-500 dark:text-zinc-400">—</p>
			) : (
				<ul className="mt-3 flex flex-wrap gap-2">
					{displayedZones.map((zone) => (
						<li key={zone.id}>
							<Label label={zone.name} color={zone.color} muted={!zone.enabled} />
						</li>
					))}
				</ul>
			)}
		</div>
	);
};

function renderComposerSection(
	section: ComposerSection,
	onTermClick?: (termId: string) => void,
	onDataTableRowClick?: (sectionId: string, rowId: string) => void,
	isEditingActive = false,
	onEditSql?: (sectionId: string, sql: string) => void,
	onEntityClick?: (focusId: string) => void,
): ReactNode {
	switch (section.type) {
		case ComposerSectionKind.TEXT_CARD:
			return (
				<div
					id={section.id === 'description' ? 'description-section' : undefined}
					className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
				>
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{section.title}
					</h2>
					<p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
						{section.body}
					</p>
				</div>
			);
		case ComposerSectionKind.TAG_LIST:
			return (
				<ReadOnlyTagList
					title={section.title}
					values={section.values}
					sectionId={section.id}
				/>
			);
		case ComposerSectionKind.INFO_GRID:
			return (
				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{section.title}
					</h2>
					<dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
						{section.items.map((item) => (
							<div
								key={item.label}
								className="rounded-lg border border-zinc-100/90 bg-zinc-50/80 p-3.5 dark:border-zinc-800 dark:bg-zinc-900/60"
							>
								<dt className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
									{item.label}
								</dt>
								<dd className="mt-1.5 text-sm text-zinc-900 dark:text-zinc-100">
									{item.value}
								</dd>
							</div>
						))}
					</dl>
				</div>
			);
		case ComposerSectionKind.DATA_TABLE: {
			const isClickable =
				section.rowIdKey != null && onDataTableRowClick != null && section.rowIdKey !== '';
			return (
				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{section.title}
					</h2>
					<Table
						className="mt-4"
						containerClassName="overflow-x-auto rounded-md border border-zinc-200/90 dark:border-zinc-700"
						layout="auto"
						minWidthClass="min-w-[28rem]"
						cellClassName="px-3 py-2"
						theadClassName="border-b border-zinc-200 bg-zinc-100/95 text-left text-xs font-semibold uppercase tracking-wide text-zinc-600 dark:border-zinc-700 dark:bg-zinc-800/90 dark:text-zinc-300"
						bodyClassName="text-zinc-800 dark:text-zinc-200"
						rowClassName="border-b border-zinc-100 transition-colors hover:bg-zinc-50/80 dark:border-zinc-800 dark:hover:bg-zinc-900/50"
						columns={section.columns.map((col) => ({
							key: col.key,
							header: col.label,
							cell: (row: Record<string, string | string[]>) => {
								const value = row[col.key];
								if (col.kind === 'tags') {
									return <LabelList values={Array.isArray(value) ? value : []} />;
								}
								const text = typeof value === 'string' ? value : '';
								if (!text) return '—';
								return col.truncate ? (
									<TruncatedText text={text} maxWidthClass={col.maxWidthClass} />
								) : (
									text
								);
							},
						}))}
						rows={section.rows}
						rowKey={(row, index) => {
							const rowId = section.rowIdKey ? row[section.rowIdKey] : undefined;
							return isClickable && typeof rowId === 'string' ? rowId : String(index);
						}}
						onRowClick={
							isClickable && section.rowIdKey
								? (row) => {
										const rowId = row[section.rowIdKey as string];
										if (rowId) {
											onDataTableRowClick(section.id, rowId as string);
										}
									}
								: undefined
						}
						emptyMessage={section.emptyMessage}
					/>
				</div>
			);
		}
		case ComposerSectionKind.LOADING_PANEL:
			return (
				<div
					className="flex min-h-[min(50dvh,420px)] flex-col items-center justify-center gap-4 rounded-lg border border-zinc-200/80 bg-white/70 px-8 py-12 dark:border-zinc-700/80 dark:bg-zinc-950/40"
					role="status"
				>
					<Spinner aria-label="Loading" />
					<p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
						{section.message}
					</p>
				</div>
			);
		case ComposerSectionKind.ZONES_CHIPS:
			return <ZonesSection section={section} />;
		case ComposerSectionKind.RELATED_TERMS_CHIPS:
			return (
				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{section.title} ({section.terms.length})
					</h2>
					{section.terms.length === 0 ? (
						<p className="mt-3 text-sm italic text-zinc-500 dark:text-zinc-400">—</p>
					) : (
						<ul className="mt-3 flex flex-wrap gap-2">
							{section.terms.map((term) => (
								<li key={term.id}>
									<Label
										label={term.name}
										title={term.description ?? undefined}
										onClick={
											onTermClick ? () => onTermClick(term.id) : undefined
										}
									/>
								</li>
							))}
						</ul>
					)}
				</div>
			);
		case ComposerSectionKind.ENTITY_CHIPS:
			return (
				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{section.title} ({section.entities.length})
					</h2>
					{section.entities.length === 0 ? (
						<p className="mt-3 text-sm italic text-zinc-500 dark:text-zinc-400">—</p>
					) : (
						<ul className="mt-3 flex flex-wrap gap-2">
							{section.entities.map((entity) => (
								<li key={entity.id}>
									<Label
										label={entity.name}
										title={entity.name}
										onClick={
											onEntityClick
												? () => onEntityClick(entity.focusId)
												: undefined
										}
									/>
								</li>
							))}
						</ul>
					)}
				</div>
			);
		case ComposerSectionKind.SQL_BLOCK:
			return (
				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					{isEditingActive && section.editable === true && (
						<div className="mb-3 flex justify-end">
							<div className="flex shrink-0 items-center gap-1">
								<button
									type="button"
									onClick={() => {
										if (onEditSql) {
											onEditSql(section.id, section.sql);
											return;
										}
										console.log('edit sql section', section.id);
									}}
									aria-label={`Edit ${section.title}`}
									title="Edit"
									className="cursor-pointer rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-[#76b900] dark:text-zinc-400 dark:hover:bg-zinc-800"
								>
									<Icon name={IconName.Pencil} className="h-4 w-4" />
								</button>
							</div>
						</div>
					)}
					{section.sql.trim() !== '' ? (
						<SqlBlock sql={section.sql} />
					) : (
						<p className="mt-3 text-sm italic text-zinc-500 dark:text-zinc-400">—</p>
					)}
				</div>
			);
		default:
			return null;
	}
}

function breadcrumbsFromPdf(headerProps: Record<string, unknown> | undefined): Breadcrumb[] {
	const raw = headerProps?.breadcrumbs;
	if (!Array.isArray(raw)) return [];
	return raw.filter(
		(b): b is Breadcrumb =>
			b !== null && typeof b === 'object' && typeof (b as Breadcrumb).name === 'string',
	);
}

export const SinglePageComposer = forwardRef<HTMLDivElement, SinglePageComposerProps>(
	function SinglePageComposer(props, ref) {
		const {
			sections,
			header,
			leftPanel,
			rightPanel,
			pdfProps,
			entityUpdatingProperties,
			isEditingMode: controlledEditingMode = false,
			onPatchEdits,
			onSave,
			onCancel,
			onDataTableRowClick,
			onEditSql,
			onSuggestDescription,
			inlineSaveSectionId,
			hideEditToolbar = false,
		} = props;
		const router = useRouter();

		const handleTermClick = useCallback(
			(termId: string) => {
				router.push(`/terms?focus=${encodeURIComponent(termId)}`);
			},
			[router],
		);

		const handleEntityClick = useCallback(
			(focusId: string) => {
				router.push(catalogPathFromFocusId(focusId));
			},
			[router],
		);

		const hh = header?.header;
		const title = (hh?.title as string) ?? 'Untitled';
		const entityId = (hh?.entityId as string) ?? '';
		const showContentHeader = hh?.showContentHeader === true;
		const titleEditable = hh?.titleEditable === true;
		const shouldAutofocusTitle = showContentHeader && titleEditable;
		const pdfPageName = (hh?.pdfProps as { pageName?: string } | undefined)?.pageName;
		const handleIsPDF = (hh?.pdfProps as { handleIsPDF?: (v: boolean) => void })?.handleIsPDF;

		const firstEditableId =
			sections.find(
				(s): s is ComposerSection =>
					isComposerSection(s) &&
					(s.type === ComposerSectionKind.TEXT_CARD ||
						s.type === ComposerSectionKind.TAG_LIST) &&
					s.editable === true,
			)?.id ?? null;
		const hasEditableSections = firstEditableId !== null;
		const [localEditingMode, setLocalEditingMode] = useState(false);
		const [saving, setSaving] = useState(false);
		const [saveError, setSaveError] = useState<string | null>(null);
		const isEditingActive = controlledEditingMode || localEditingMode;

		const titleInputRef = useRef<HTMLInputElement>(null);
		const pendingEditsRef = useRef<Record<string, ComposerEditValue>>({});

		// Reset the save error when an *externally* controlled `isEditingMode` prop
		// flips to true (e.g. a parent toggling edit mode) — mirrors what the
		// internal "Edit" button already does for `localEditingMode`. Adjusted
		// during render (not in an effect) per
		// https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes.
		const [prevIsEditingActive, setPrevIsEditingActive] = useState(isEditingActive);
		if (isEditingActive !== prevIsEditingActive) {
			setPrevIsEditingActive(isEditingActive);
			if (isEditingActive) {
				setSaveError(null);
			}
		}

		// Refs can't be mutated during render, so the pending-edits reset for
		// that same transition lives in its own effect (no setState here, so
		// it doesn't trip the "no setState in effects" rule above).
		useEffect(() => {
			if (isEditingActive) {
				pendingEditsRef.current = {};
			}
		}, [isEditingActive]);

		useEffect(() => {
			if (isEditingActive && shouldAutofocusTitle) {
				titleInputRef.current?.focus();
			}
		}, [isEditingActive, shouldAutofocusTitle]);

		const handleSave = async () => {
			const edits = pendingEditsRef.current;
			if (saving) return;

			const hasPendingEdits = Object.keys(edits).length > 0;
			const usesDefaultCatalogSave = onPatchEdits == null;
			const hasNothingToSave = !entityId || !hasPendingEdits;

			if (usesDefaultCatalogSave && hasNothingToSave) {
				setLocalEditingMode(false);
				setSaveError(null);
				onSave?.(edits);
				return;
			}

			setSaving(true);
			setSaveError(null);
			const res = onPatchEdits
				? await onPatchEdits(edits)
				: await datasources.updateNode(entityId, edits as NodePatch);
			setSaving(false);

			if (res.error) {
				setSaveError(res.message ?? 'Failed to save changes');
				return;
			}

			setLocalEditingMode(false);
			onSave?.(edits);
		};

		const pdfHeader = pdfProps?.headerProps;
		const breadcrumbs = breadcrumbsFromPdf(pdfHeader);
		const isPDFView = pdfProps?.isPDFView ?? false;

		const gridTemplate =
			leftPanel && rightPanel
				? `[full-start] ${leftPanel.width} [main-start] 1fr [main-end] ${rightPanel.width} [full-end]`
				: leftPanel
					? `[full-start] ${leftPanel.width} [main-start] 1fr [main-end full-end]`
					: rightPanel
						? `[full-start main-start] 1fr [main-end] ${rightPanel.width} [full-end]`
						: `[full-start main-start] 1fr [main-end full-end]`;

		if (isPDFView) {
			return (
				<div
					ref={ref}
					className="flex min-h-[50vh] w-full flex-col gap-6 rounded-lg border border-zinc-200 bg-white p-8 text-zinc-900 shadow-sm dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-50"
				>
					<div className="border-b border-zinc-200 pb-4 dark:border-zinc-700">
						<p className="text-xs font-medium uppercase tracking-wide text-zinc-500">
							PDF preview
							{pdfPageName ? (
								<span className="ml-2 font-normal normal-case text-zinc-600 dark:text-zinc-400">
									{pdfPageName}
								</span>
							) : null}
						</p>
						<div className="mt-2 flex flex-wrap gap-1 text-sm text-zinc-600 dark:text-zinc-400">
							{breadcrumbs.map((b, i) => (
								<span key={b.id ?? `${b.path}-${i}`}>
									{i > 0 ? <span className="mx-1 text-zinc-400">/</span> : null}
									{b.name}
								</span>
							))}
						</div>
						<h1 className="mt-3 text-2xl font-semibold tracking-tight">{title}</h1>
					</div>
					<ol className="list-decimal space-y-4 pl-5 text-sm">
						{sections.map((s, i) => (
							<li key={i} className="text-zinc-700 dark:text-zinc-300">
								<span className="font-medium">
									{isComposerSection(s) ? composerSectionHeading(s) : 'Section'}
								</span>
							</li>
						))}
					</ol>
					{handleIsPDF ? (
						<button
							type="button"
							className="self-start rounded-md bg-zinc-200 px-3 py-1.5 text-sm font-medium text-zinc-900 hover:bg-zinc-300 dark:bg-zinc-800 dark:text-zinc-100 dark:hover:bg-zinc-700"
							onClick={() => handleIsPDF(false)}
						>
							Close PDF preview
						</button>
					) : null}
				</div>
			);
		}

		return (
			<div
				ref={ref}
				className="box-border flex h-full min-h-0 w-full min-w-0 flex-1 flex-col gap-0 overflow-hidden bg-white dark:bg-zinc-950"
			>
				{header?.errorBanner ? (
					<div className="border-b border-amber-200/90 bg-amber-50 px-7 py-4 text-sm text-amber-950 sm:px-10 dark:border-amber-900/50 dark:bg-amber-950/40 dark:text-amber-100">
						{String(header.errorBanner)}
					</div>
				) : null}

				{!hideEditToolbar &&
				((hasEditableSections && entityId && !isEditingActive) || isEditingActive) ? (
					<div className="flex shrink-0 items-center justify-end px-7 py-2 sm:px-10">
						{hasEditableSections && entityId && !isEditingActive && (
							<button
								type="button"
								onClick={() => {
									pendingEditsRef.current = {};
									setSaveError(null);
									setLocalEditingMode(true);
								}}
								className="flex items-center gap-1.5 rounded-lg bg-[#76b900] px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-[#6aa500]"
							>
								<Icon name={IconName.Pencil} className="h-3.5 w-3.5" />
								Edit
							</button>
						)}
						{isEditingActive && (
							<div className="flex items-center gap-2">
								<button
									type="button"
									disabled={saving}
									onClick={() => {
										if (saving) return;
										setLocalEditingMode(false);
										setSaveError(null);
										onCancel?.();
									}}
									className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800"
								>
									Cancel
								</button>
								{inlineSaveSectionId == null && (
									<button
										type="button"
										disabled={saving}
										onClick={() => {
											void handleSave();
										}}
										className="flex items-center gap-1.5 rounded-lg bg-[#76b900] px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-[#6aa500] disabled:cursor-not-allowed disabled:opacity-70"
									>
										{saving ? (
											<>
												<Spinner
													aria-label="Saving"
													className="h-3.5 w-3.5"
												/>
												Saving…
											</>
										) : (
											'Save'
										)}
									</button>
								)}
							</div>
						)}
					</div>
				) : null}

				<div
					className="grid min-h-0 w-full min-w-0 flex-1 gap-0 overflow-hidden"
					style={{ gridTemplateColumns: gridTemplate }}
				>
					{leftPanel ? (
						<aside className="flex min-h-0 min-w-0 flex-col border-r border-zinc-200/80 bg-gradient-to-b from-zinc-50/95 to-white dark:border-zinc-700/80 dark:from-zinc-950 dark:to-zinc-950/90">
							{leftPanel.slot ? (
								<div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden p-0">
									{leftPanel.slot}
								</div>
							) : (
								<div className="px-5 py-4 text-sm text-zinc-600 sm:px-6 dark:text-zinc-400">
									<p className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
										Left panel
									</p>
									<p className="text-xs text-zinc-500">
										{leftPanel.bulks.length} bulk(s) · width {leftPanel.width}
									</p>
								</div>
							)}
						</aside>
					) : null}

					<main className="min-h-0 min-w-0 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]">
						{sections.length === 0 ? (
							<div className="flex min-h-[14rem] flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-zinc-300/90 bg-white/70 px-8 py-12 text-center dark:border-zinc-600 dark:bg-zinc-900/30">
								<div
									className="flex h-12 w-12 items-center justify-center rounded-lg bg-[#76b900]/15 text-xl"
									aria-hidden
								>
									◇
								</div>
								<p className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
									Nothing selected yet
								</p>
								<p className="max-w-sm text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
									Pick a database, schema, table, column, or field in the explorer
									to load metadata, descriptions, and related entities.
								</p>
							</div>
						) : (
							<div className="space-y-5">
								{showContentHeader ? (
									<div>
										{isEditingActive && titleEditable ? (
											<input
												ref={titleInputRef}
												key={title}
												type="text"
												autoFocus
												defaultValue={title}
												onChange={(e) => {
													pendingEditsRef.current.name = e.target.value;
												}}
												className="w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-2xl font-semibold tracking-tight text-zinc-900 outline-none transition-colors focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100"
												aria-label="Name"
											/>
										) : (
											<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
												{title}
											</h1>
										)}
									</div>
								) : null}
								{sections.map((section, i) => {
									if (!isComposerSection(section)) {
										return (
											<section
												key={i}
												className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
											>
												<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
													{typeof section === 'object' &&
													section !== null &&
													'title' in section &&
													typeof (section as { title: unknown }).title ===
														'string'
														? (section as { title: string }).title
														: 'Section'}
												</h2>
												<pre className="mt-2 max-h-40 overflow-auto rounded bg-zinc-100 p-2 text-xs text-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
													{JSON.stringify(section, null, 2)}
												</pre>
											</section>
										);
									}

									const isEditableTextCard =
										isEditingActive &&
										section.type === ComposerSectionKind.TEXT_CARD &&
										section.editable === true;
									const isEditableTagList =
										isEditingActive &&
										section.type === ComposerSectionKind.TAG_LIST &&
										section.editable === true;
									return (
										<div key={section.id}>
											{isEditableTextCard ? (
												<EditableTextCard
													section={section}
													autoFocus={
														!shouldAutofocusTitle &&
														section.id === firstEditableId
													}
													onChange={(id, val) => {
														pendingEditsRef.current[id] = val;
													}}
													onSave={
														section.id === inlineSaveSectionId
															? () => {
																	void handleSave();
																}
															: undefined
													}
													saving={saving}
													onSuggest={
														section.suggestable === true &&
														onSuggestDescription != null
															? () => onSuggestDescription(section.id)
															: undefined
													}
												/>
											) : isEditableTagList ? (
												<EditableTagListCard
													section={section}
													autoFocus={
														!shouldAutofocusTitle &&
														section.id === firstEditableId
													}
													onChange={(id, val) => {
														pendingEditsRef.current[id] = val;
													}}
												/>
											) : (
												renderComposerSection(
													section,
													handleTermClick,
													onDataTableRowClick,
													isEditingActive,
													onEditSql,
													handleEntityClick,
												)
											)}
										</div>
									);
								})}
							</div>
						)}

						{entityUpdatingProperties &&
						Object.keys(entityUpdatingProperties).length > 0 ? (
							<div className="mt-6 rounded-lg border border-dashed border-zinc-300 p-3 dark:border-zinc-600">
								<p className="mb-2 text-xs font-semibold uppercase text-zinc-500">
									Entity updates (preview)
								</p>
								<ul className="space-y-1 text-xs text-zinc-600 dark:text-zinc-400">
									{Object.entries(entityUpdatingProperties).map(([k, v]) => (
										<li key={k}>
											<span className="font-mono text-zinc-800 dark:text-zinc-200">
												{k}
											</span>
											: {Array.isArray(v) ? v.join(', ') : v}
										</li>
									))}
								</ul>
							</div>
						) : null}
					</main>

					{rightPanel ? (
						<aside className="flex min-h-0 min-w-0 flex-col border-l border-zinc-200/80 bg-gradient-to-b from-zinc-50/95 to-white dark:border-zinc-700/80 dark:from-zinc-950 dark:to-zinc-950/90">
							{rightPanel.slot ? (
								<div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden px-4 py-4 sm:px-5">
									{rightPanel.slot}
								</div>
							) : (
								<div className="px-5 py-4 text-sm text-zinc-600 sm:px-6 dark:text-zinc-400">
									<p className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
										Right panel
									</p>
									<p className="text-xs text-zinc-500">
										{rightPanel.bulks.length} bulk(s) · width {rightPanel.width}
									</p>
								</div>
							)}
						</aside>
					) : null}
				</div>

				<Toast
					open={saveError != null}
					message={saveError ?? ''}
					variant="error"
					onClose={() => setSaveError(null)}
				/>
			</div>
		);
	},
);
