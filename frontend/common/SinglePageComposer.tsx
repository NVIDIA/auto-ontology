// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { forwardRef, useEffect, useRef, useState, type ReactNode } from 'react';
import { Spinner } from '@nvidia/foundations-react-core';
import type { Breadcrumb } from '@/types/breadcrumbs';
import { ComposerSectionKind } from '@/enums/datasources';
import { isComposerSection, type ComposerSection } from '@/types/composer-section';
import { Icon, IconName } from '@/components/icons';
import { TagInput } from '@/components/TagInput';
import { Table } from '@/components/Table';
import { datasources } from '@/api/datasources';
import type { NodePatch } from '@/api/types';
import { Toast } from '@/components/Toast';

export type ComposerEditValue = string | string[];

function zoneChipStyle(color: string): React.CSSProperties {
	return { backgroundColor: `${color}26`, color, borderColor: `${color}60` };
}

const ZoneChip = ({ name, color }: { name: string; color: string | null }) => (
	<span
		className="inline-flex items-center rounded-full border border-zinc-200 bg-zinc-100 px-3 py-1 text-xs font-medium text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
		style={color ? zoneChipStyle(color) : undefined}
	>
		{name}
	</span>
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
	isEditing?: boolean;
	onSave?: (edits: Record<string, ComposerEditValue>) => void;
	onCancel?: () => void;
};

function composerSectionHeading(section: ComposerSection): string {
	switch (section.type) {
		case ComposerSectionKind.LOADING_PANEL:
			return section.message;
		case ComposerSectionKind.ZONES_CHIPS:
			return `${section.title} (${section.zones.length})`;
		default:
			return section.title;
	}
}

const EditableTextCard = ({
	section,
	onChange,
	autoFocus = false,
}: {
	section: { id: string; title: string; body: string };
	onChange: (sectionId: string, value: string) => void;
	autoFocus?: boolean;
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
}) => (
	<div
		id={sectionId === 'sample_values' ? 'sample-values-section' : undefined}
		className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]"
	>
		<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</h2>
		{values.length === 0 ? (
			<p className="mt-3 text-sm italic text-zinc-500 dark:text-zinc-400">—</p>
		) : (
			<ul className="mt-3 flex flex-wrap gap-1.5">
				{values.map((v, i) => (
					<li
						key={`${v}-${i}`}
						className="inline-flex max-w-full items-center rounded-md border border-zinc-200 bg-zinc-50 px-2 py-0.5 text-xs font-medium text-zinc-700 dark:border-zinc-700 dark:bg-zinc-900/60 dark:text-zinc-200"
					>
						<span className="max-w-[24rem] truncate" title={v}>
							{v}
						</span>
					</li>
				))}
			</ul>
		)}
	</div>
);

function renderComposerSection(section: ComposerSection): ReactNode {
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
		case ComposerSectionKind.DATA_TABLE:
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
							cell: (row: Record<string, string>) => row[col.key] || '—',
						}))}
						rows={section.rows}
						rowKey={(_, index) => String(index)}
					/>
				</div>
			);
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
			return (
				<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{section.title} ({section.zones.length})
					</h2>
					{section.zones.length === 0 ? (
						<p className="mt-3 text-sm italic text-zinc-500 dark:text-zinc-400">—</p>
					) : (
						<ul className="mt-3 flex flex-wrap gap-2">
							{section.zones.map((zone) => (
								<li key={zone.id}>
									<ZoneChip name={zone.name} color={zone.color} />
								</li>
							))}
						</ul>
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
			isEditing,
			onSave,
			onCancel,
		} = props;
		const hh = header?.header;
		const title = (hh?.title as string) ?? 'Untitled';
		const entityId = (hh?.entityId as string) ?? '';
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
		const [localIsEditing, setLocalIsEditing] = useState(false);
		const [saving, setSaving] = useState(false);
		const [saveError, setSaveError] = useState<string | null>(null);
		const isEditingActive = isEditing || localIsEditing;

		const pendingEditsRef = useRef<Record<string, ComposerEditValue>>({});

		const handleSave = async () => {
			const edits = pendingEditsRef.current;
			if (saving) return;

			if (!entityId || Object.keys(edits).length === 0) {
				setLocalIsEditing(false);
				setSaveError(null);
				onSave?.(edits);
				return;
			}

			setSaving(true);
			setSaveError(null);
			const res = await datasources.updateNode(entityId, edits as NodePatch);
			setSaving(false);

			if (res.error === true) {
				setSaveError(res.message ?? 'Failed to save changes');
				return;
			}

			setLocalIsEditing(false);
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

				{(hasEditableSections && entityId && !isEditingActive) || isEditingActive ? (
					<div className="flex shrink-0 items-center justify-end px-7 py-2 sm:px-10">
						{hasEditableSections && entityId && !isEditingActive && (
							<button
								type="button"
								onClick={() => {
									pendingEditsRef.current = {};
									setSaveError(null);
									setLocalIsEditing(true);
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
										setLocalIsEditing(false);
										setSaveError(null);
										onCancel?.();
									}}
									className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-600 dark:text-zinc-300 dark:hover:bg-zinc-800"
								>
									Cancel
								</button>
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
													autoFocus={section.id === firstEditableId}
													onChange={(id, val) => {
														pendingEditsRef.current[id] = val;
													}}
												/>
											) : isEditableTagList ? (
												<EditableTagListCard
													section={section}
													autoFocus={section.id === firstEditableId}
													onChange={(id, val) => {
														pendingEditsRef.current[id] = val;
													}}
												/>
											) : (
												renderComposerSection(section)
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
