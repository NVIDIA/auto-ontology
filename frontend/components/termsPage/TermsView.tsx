// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import { Placeholders } from '@/assets/images/placeholders';
import { Icon, IconName } from '@/components/icons';
import { ModalCreateNewItem } from '@/components/ModalCreateNewItem';
import { SqlEditor } from '@/components/SqlBlock';
import { TagInput } from '@/components/TagInput';
import { termsApi } from '@/api/terms';
import { ComposerSectionKind } from '@/enums/datasources';
import { SinglePageView, type SinglePageFormat } from '@/components/SinglePageView';
import type { Term, TermAttribute } from '@/types/terms';

const FIELD_CLASS =
	'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500';

const Label = ({ children }: { children: React.ReactNode }) => (
	<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
		{children}
	</label>
);

const SectionHeading = ({ children }: { children: React.ReactNode }) => (
	<p className="text-xs font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-500">
		{children}
	</p>
);

type TermCardProps = {
	term: Term;
	attributes: TermAttribute[];
	relatedCount: number;
	onClick: (term: Term) => void;
};

const TermCard = ({ term, attributes, relatedCount, onClick }: TermCardProps) => (
	<li
		role="button"
		tabIndex={0}
		onClick={() => onClick(term)}
		onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onClick(term)}
		className="cursor-pointer rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
	>
		{/* Card header */}
		<div className="flex items-start justify-between gap-3">
			<div>
				<h2 className="text-base font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					{term.name}
				</h2>
				{term.synonyms.length > 0 && (
					<p className="mt-0.5 text-xs text-zinc-400 dark:text-zinc-500">
						{term.synonyms.join(', ')}
					</p>
				)}
			</div>
		</div>

		{term.description != null && term.description.trim() !== '' ? (
			<p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">{term.description}</p>
		) : (
			<p className="mt-2 text-sm italic text-zinc-400 dark:text-zinc-500">
				No Description Available
			</p>
		)}

		{/* Two-column section */}
		<div className="mt-4 grid grid-cols-2 gap-0 overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-700">
			{/* Column Attributes */}
			<div className="border-r border-zinc-200 dark:border-zinc-700">
				<div className="border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
					<span className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
						Column Attributes
					</span>
				</div>
				<div className="flex items-center">
					<span className="flex w-full items-center justify-between px-4 py-3 text-sm text-zinc-700 dark:text-zinc-300">
						<span>Column Attributes</span>
						<span className="ml-1.5 font-medium text-zinc-900 dark:text-zinc-100">
							{attributes.length}
						</span>
					</span>
				</div>
			</div>

			{/* Related Terms */}
			<div>
				<div className="border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-700">
					<span className="text-xs font-semibold text-zinc-500 dark:text-zinc-400">
						Related Terms
					</span>
				</div>
				<div className="flex items-center justify-between px-4 py-3 text-sm text-zinc-700 dark:text-zinc-300">
					<span>Related Terms</span>
					<span className="font-medium text-zinc-900 dark:text-zinc-100">
						{relatedCount}
					</span>
				</div>
			</div>
		</div>
	</li>
);

export const TermsView = () => {
	const router = useRouter();
	const searchParams = useSearchParams();
	const focusId = searchParams.get('focus');

	const [terms, setTerms] = useState<Term[]>([]);
	const [attrs, setAttrs] = useState<TermAttribute[]>([]);
	const [relatedCountsMap, setRelatedCountsMap] = useState<Map<string, number>>(new Map());
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const [modalOpen, setModalOpen] = useState(false);
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [synonyms, setSynonyms] = useState<string[]>([]);
	const [sqlAttrName, setSqlAttrName] = useState('');
	const [sqlAttrDescription, setSqlAttrDescription] = useState('');
	const [sql, setSql] = useState('');

	const trimmedName = name.trim();
	const canSubmit = trimmedName.length > 0;

	useEffect(() => {
		let cancelled = false;

		(async () => {
			setLoading(true);
			const [termsRes, attrsRes, countsRes] = await Promise.all([
				termsApi.list(),
				termsApi.listColumnAttributes(),
				termsApi.listRelatedCounts(),
			]);
			if (cancelled) return;

			if (termsRes.error === true) {
				setError(termsRes.message ?? 'Failed to load terms');
				setTerms([]);
			} else {
				setError(null);
				setTerms(termsRes.data ?? []);
			}

			if (attrsRes.error !== true) {
				setAttrs(attrsRes.data ?? []);
			}

			if (countsRes.error !== true) {
				const map = new Map<string, number>();
				for (const { term_id, count } of countsRes.data ?? []) {
					map.set(term_id, count);
				}
				setRelatedCountsMap(map);
			}

			setLoading(false);
		})();

		return () => {
			cancelled = true;
		};
	}, []);

	const attrsByTerm = useMemo(() => {
		const map = new Map<string, TermAttribute[]>();
		for (const attr of attrs) {
			const list = map.get(attr.term_name) ?? [];
			list.push(attr);
			map.set(attr.term_name, list);
		}
		return map;
	}, [attrs]);

	const handleCardClick = useCallback(
		(term: Term) => {
			router.push(`/terms?focus=${encodeURIComponent(term.id)}`);
		},
		[router],
	);

	const handleBack = useCallback(() => {
		router.push('/terms');
	}, [router]);

	const getSinglePage = useCallback(async (termId: string): Promise<SinglePageFormat> => {
		const [res, attrsRes, relatedRes] = await Promise.all([
			termsApi.get(termId),
			termsApi.getColumnAttributes(termId),
			termsApi.getRelatedTerms(termId),
		]);
		if (res.error === true || !res.data) {
			return {
				sections: [],
				header: { header: { title: 'Term not found', withBorder: true } },
			};
		}
		const term = res.data;
		const termAttrs = attrsRes.error !== true ? (attrsRes.data ?? []) : [];
		const relatedTerms = relatedRes.error !== true ? (relatedRes.data ?? []) : [];

		return {
			header: {
				header: {
					title: term.name,
					withBorder: true,
				},
			},
			sections: [
				{
					type: ComposerSectionKind.TEXT_CARD,
					id: 'description',
					title: 'Description',
					body: term.description ?? '',
				},
				{
					type: ComposerSectionKind.TEXT_CARD,
					id: 'entities',
					title: 'Entities',
					body: `Tables (${term.table_count})`,
				},
				{
					type: ComposerSectionKind.ZONES_CHIPS,
					id: 'zones',
					title: 'Zones',
					zones: term.zones.map((z) => ({
						id: z.id,
						name: z.name,
						color: z.color,
					})),
				},
				{
					type: ComposerSectionKind.RELATED_TERMS_CHIPS,
					id: 'related_terms',
					title: 'Related Terms',
					terms: relatedTerms.map((t) => ({
						id: t.id,
						name: t.name,
						description: t.description,
					})),
				},
				{
					type: ComposerSectionKind.DATA_TABLE,
					id: 'column_attributes',
					title: 'Column Attributes',
					columns: [{ key: 'name', label: 'Attribute Name' }],
					rows: termAttrs.map((attr) => ({
						name: attr.name,
					})),
				},
			],
		};
	}, []);

	const focusedTerm = focusId != null ? (terms.find((t) => t.id === focusId) ?? null) : null;

	const openModal = () => {
		setName('');
		setDescription('');
		setSynonyms([]);
		setSqlAttrName('');
		setSqlAttrDescription('');
		setSql('');
		setModalOpen(true);
	};

	const handleClose = () => setModalOpen(false);

	const handleSubmit = () => {
		console.log('Create new term', {
			name: trimmedName,
			description: description.trim(),
			synonyms,
			sqlAttribute: {
				name: sqlAttrName.trim(),
				description: sqlAttrDescription.trim(),
				sql: sql.trim(),
			},
		});
		setModalOpen(false);
	};

	if (focusId != null) {
		const termTitle = focusedTerm?.name ?? focusId;
		return (
			<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
				<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
					<button
						type="button"
						onClick={handleBack}
						className="flex cursor-pointer items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
						aria-label="Back to terms list"
					>
						<Icon name={IconName.Terms} className="h-4 w-4" />
						Terms
					</button>
					<span className="text-zinc-300 dark:text-zinc-600">/</span>
					<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
						{termTitle}
					</span>
				</header>
				<main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
					<SinglePageView
						dataId={focusId}
						title={termTitle}
						getSinglePage={getSinglePage}
					/>
				</main>
			</div>
		);
	}

	return (
		<div className="flex h-full w-full flex-col bg-white dark:bg-zinc-950">
			<header className="flex items-center gap-3 border-b border-zinc-200 px-6 py-4 dark:border-zinc-800">
				<Icon name={IconName.Terms} className="h-5 w-5 text-[#76b900]" />
				<h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-100">
					Terms
				</h1>
			</header>

			<div className="flex-1 overflow-y-auto px-6 py-6">
				{loading && (
					<div className="flex h-full items-center justify-center">
						<div
							className="h-10 w-10 animate-spin rounded-full border-2 border-zinc-200 border-t-[#76b900] dark:border-zinc-700"
							role="status"
							aria-label="Loading terms"
						/>
					</div>
				)}

				{!loading && error != null && (
					<div className="mx-auto max-w-lg rounded-2xl border border-red-200/80 bg-white/90 px-8 py-10 text-center shadow-xl shadow-red-100/50 dark:border-red-900/50 dark:bg-zinc-950/80 dark:shadow-none">
						<h2 className="text-lg font-semibold tracking-tight text-red-800 dark:text-red-300">
							Couldn&apos;t load terms
						</h2>
						<pre className="mt-4 max-w-full overflow-x-auto rounded-lg border border-red-100 bg-red-50/80 p-3 text-left text-xs text-red-900/80 dark:border-red-900/40 dark:bg-red-950/40 dark:text-red-200">
							{error}
						</pre>
					</div>
				)}

				{!loading && error == null && terms.length === 0 && (
					<div className="flex h-full min-h-[40dvh] flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-zinc-300/80 bg-white/60 p-12 text-center dark:border-zinc-600 dark:bg-zinc-950/40">
						<Placeholders.NoTerms />
						<p className="mt-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
							No Terms Created Yet
						</p>
						<button
							type="button"
							onClick={openModal}
							className="mt-1 flex cursor-pointer items-center gap-2 rounded-lg bg-[#76b900] px-4 py-2.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#6aa500]"
						>
							<Icon name={IconName.Terms} className="h-4 w-4" />
							Create new term
						</button>
					</div>
				)}

				{!loading && error == null && terms.length > 0 && (
					<ul className="flex flex-col gap-4">
						{terms.map((term) => (
							<TermCard
								key={term.id}
								term={term}
								attributes={attrsByTerm.get(term.name) ?? []}
								relatedCount={relatedCountsMap.get(term.id) ?? 0}
								onClick={handleCardClick}
							/>
						))}
					</ul>
				)}
			</div>

			<ModalCreateNewItem
				open={modalOpen}
				onClose={handleClose}
				title="Create New Term"
				submitLabel="Save"
				onSubmit={handleSubmit}
				canSubmit={canSubmit}
				className="min-h-[500px] w-[640px] max-w-full"
			>
				<div>
					<Label>Name</Label>
					<input
						type="text"
						value={name}
						onChange={(e) => setName(e.target.value)}
						placeholder="Term name"
						className={FIELD_CLASS}
					/>
				</div>

				<div>
					<Label>Description</Label>
					<textarea
						value={description}
						onChange={(e) => setDescription(e.target.value)}
						placeholder="Short description"
						rows={3}
						className={`${FIELD_CLASS} resize-y`}
					/>
				</div>

				<div>
					<Label>Synonyms</Label>
					<TagInput
						value={synonyms}
						onChange={setSynonyms}
						placeholder="Type and press Enter"
						ariaLabel="Synonyms"
					/>
				</div>

				<div className="pt-2">
					<SectionHeading>SQL Attribute</SectionHeading>
				</div>

				<div>
					<Label>Attribute Name</Label>
					<input
						type="text"
						value={sqlAttrName}
						onChange={(e) => setSqlAttrName(e.target.value)}
						placeholder="Attribute name"
						className={FIELD_CLASS}
					/>
				</div>

				<div>
					<Label>Attribute Description</Label>
					<textarea
						value={sqlAttrDescription}
						onChange={(e) => setSqlAttrDescription(e.target.value)}
						placeholder="Attribute description"
						rows={2}
						className={`${FIELD_CLASS} resize-y`}
					/>
				</div>

				<SqlEditor
					value={sql}
					onChange={setSql}
					label="SQL"
					placeholder="SELECT ..."
					rows={6}
				/>
			</ModalCreateNewItem>
		</div>
	);
};
