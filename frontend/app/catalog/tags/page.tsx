// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { om } from '@/api/openmetadata';
import type { OmClassification, OmTagDef } from '@/types/openmetadata';
import { TagPill } from '@/components/catalog/TagPill';
import { Panel } from '@/components/catalog/ui';

type ClassWithTags = OmClassification & { tags: OmTagDef[] };

export default function TagsPage() {
	const [data, setData] = useState<ClassWithTags[] | null>(null);
	const [err, setErr] = useState<string | null>(null);
	const [pickedName, setPickedName] = useState<string | null>(null);
	const [creatingFor, setCreatingFor] = useState<string | null>(null);
	const [newClassOpen, setNewClassOpen] = useState(false);

	const [reloadKey, setReloadKey] = useState(0);
	const reload = useCallback(() => setReloadKey((k) => k + 1), []);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const cls = await om.classifications.list({ limit: 100 });
				const all = await Promise.all(
					cls.data.map(async (c) => {
						try {
							const tg = await om.tags.list({ parent: c.name, limit: 200 });
							return { ...c, tags: tg.data } as ClassWithTags;
						} catch {
							return { ...c, tags: [] } as ClassWithTags;
						}
					}),
				);
				if (!cancelled) setData(all);
			} catch (e) {
				if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
			}
		})();
		return () => {
			cancelled = true;
		};
	}, [reloadKey]);

	// Effective selection: explicit user pick, falling back to the first
	// classification once data loads. Derived (no extra state needed).
	const selected = pickedName ?? data?.[0]?.name ?? null;
	const setSelected = setPickedName;

	const current = useMemo(
		() => (data ?? []).find((c) => c.name === selected) ?? null,
		[data, selected],
	);

	const totalTags = useMemo(
		() => (data ?? []).reduce((acc, c) => acc + c.tags.length, 0),
		[data],
	);

	return (
		<div className="mx-auto w-full max-w-7xl px-6 py-8">
			<header className="mb-6 flex flex-wrap items-end justify-between gap-4">
				<div>
					<h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
						Tags
						{data != null ? (
							<span className="ml-2 rounded-full bg-zinc-100 px-2 py-0.5 align-middle text-sm font-medium tabular-nums text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
								{totalTags}
							</span>
						) : null}
					</h1>
					<p className="mt-1 max-w-2xl text-sm text-zinc-600 dark:text-zinc-400">
						Classifications group related tags. Use them to label tables and columns
						with custom themes (e.g. <span className="font-medium">Sensitivity</span>)
						or rely on the built-in <span className="font-medium">PII</span> system
						classification populated by Auto-Classification.
					</p>
				</div>
				<button
					type="button"
					onClick={() => setNewClassOpen(true)}
					className="rounded-lg bg-[#76b900] px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-[#5fa000]"
				>
					+ New classification
				</button>
			</header>

			{err ? (
				<div className="mb-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
					{err}
				</div>
			) : null}

			<div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
				<Panel title="Classifications" className="self-start">
					{data == null ? (
						<SkeletonList />
					) : (
						<ul className="p-2">
							{data.map((c) => (
								<li key={c.name}>
									<button
										type="button"
										onClick={() => setSelected(c.name)}
										className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm transition-colors ${
											c.name === selected
												? 'bg-[#76b900]/10 text-[#76b900]'
												: 'text-zinc-700 hover:bg-zinc-50 dark:text-zinc-300 dark:hover:bg-zinc-800'
										}`}
									>
										<span className="flex items-center gap-2">
											<span className="font-medium">{c.name}</span>
											{c.provider === 'system' ? (
												<span className="rounded bg-zinc-100 px-1 py-px text-[10px] font-medium uppercase tracking-wider text-zinc-500 dark:bg-zinc-800">
													system
												</span>
											) : null}
										</span>
										<span className="text-xs tabular-nums text-zinc-400">
											{c.tags.length}
										</span>
									</button>
								</li>
							))}
						</ul>
					)}
				</Panel>

				{current == null ? (
					<Panel>
						<p className="px-6 py-6 text-sm text-zinc-500">Select a classification…</p>
					</Panel>
				) : (
					<Panel
						title={current.name}
						action={
							current.provider === 'system' ? (
								<span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 dark:bg-zinc-800">
									system
								</span>
							) : (
								<button
									type="button"
									onClick={() => setCreatingFor(current.name)}
									className="rounded-lg border border-zinc-300 px-3 py-1 text-xs font-medium normal-case tracking-normal text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
								>
									+ Add tag
								</button>
							)
						}
					>
						<div className="p-5">
							{current.description ? (
								<p className="mb-4 max-w-3xl border-b border-zinc-100 pb-4 text-sm text-zinc-600 dark:border-zinc-800 dark:text-zinc-400">
									{current.description}
								</p>
							) : null}

							{current.tags.length === 0 ? (
								<p className="rounded-lg border border-dashed border-zinc-300 p-8 text-center text-sm text-zinc-500 dark:border-zinc-700">
									No tags in this classification yet.
								</p>
							) : (
								<ul className="space-y-2">
									{current.tags.map((t) => (
										<li
											key={t.fullyQualifiedName}
											className="flex items-start justify-between gap-4 rounded-lg border border-zinc-100 p-3 dark:border-zinc-800"
										>
											<div className="min-w-0">
												<div className="flex items-center gap-2">
													<TagPill
														label={{
															tagFQN: t.fullyQualifiedName,
															source: 'Classification',
															state: 'Confirmed',
															labelType: 'Manual',
														}}
													/>
													{t.mutuallyExclusive ? (
														<span className="rounded bg-violet-100 px-1.5 py-0 text-[10px] font-medium text-violet-700 dark:bg-violet-950/40 dark:text-violet-300">
															exclusive
														</span>
													) : null}
												</div>
												{t.description ? (
													<p className="mt-1.5 text-xs text-zinc-600 dark:text-zinc-400">
														{t.description}
													</p>
												) : null}
											</div>
											<code className="shrink-0 text-[10px] text-zinc-400">
												{t.fullyQualifiedName}
											</code>
										</li>
									))}
								</ul>
							)}
						</div>
					</Panel>
				)}
			</div>

			{creatingFor != null ? (
				<NewTagModal
					classification={creatingFor}
					onClose={() => setCreatingFor(null)}
					onCreated={() => {
						setCreatingFor(null);
						reload();
					}}
				/>
			) : null}
			{newClassOpen ? (
				<NewClassificationModal
					onClose={() => setNewClassOpen(false)}
					onCreated={(name) => {
						setNewClassOpen(false);
						setSelected(name);
						reload();
					}}
				/>
			) : null}
		</div>
	);
}

const SkeletonList = () => (
	<ul className="space-y-1.5 p-1">
		{Array.from({ length: 6 }).map((_, i) => (
			<li key={i} className="h-8 animate-pulse rounded-lg bg-zinc-100 dark:bg-zinc-800" />
		))}
	</ul>
);

const Modal = ({
	title,
	onClose,
	children,
}: {
	title: string;
	onClose: () => void;
	children: React.ReactNode;
}) => (
	<div
		className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/40 backdrop-blur-sm"
		role="dialog"
		aria-modal
	>
		<div className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-6 shadow-2xl dark:border-zinc-800 dark:bg-zinc-900">
			<div className="mb-4 flex items-center justify-between">
				<h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
					{title}
				</h3>
				<button
					type="button"
					onClick={onClose}
					className="rounded text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300"
					aria-label="Close"
				>
					✕
				</button>
			</div>
			{children}
		</div>
	</div>
);

const NewClassificationModal = ({
	onClose,
	onCreated,
}: {
	onClose: () => void;
	onCreated: (name: string) => void;
}) => {
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [busy, setBusy] = useState(false);
	const [err, setErr] = useState<string | null>(null);

	return (
		<Modal title="New classification" onClose={onClose}>
			<form
				onSubmit={async (e) => {
					e.preventDefault();
					setBusy(true);
					setErr(null);
					try {
						await om.classifications.create({ name, description });
						onCreated(name);
					} catch (ex) {
						setErr(ex instanceof Error ? ex.message : String(ex));
					} finally {
						setBusy(false);
					}
				}}
				className="space-y-3"
			>
				<Field label="Name" required>
					<input
						value={name}
						onChange={(e) => setName(e.target.value)}
						required
						pattern="[A-Za-z0-9_\-]+"
						title="Letters, digits, underscore, hyphen"
						className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
					/>
				</Field>
				<Field label="Description">
					<textarea
						value={description}
						onChange={(e) => setDescription(e.target.value)}
						rows={3}
						className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
					/>
				</Field>
				{err ? <p className="text-xs text-red-600">{err}</p> : null}
				<div className="flex justify-end gap-2 pt-2">
					<button
						type="button"
						onClick={onClose}
						className="rounded-lg px-3 py-1.5 text-sm text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
					>
						Cancel
					</button>
					<button
						type="submit"
						disabled={busy}
						className="rounded-lg bg-[#76b900] px-3 py-1.5 text-sm font-medium text-white hover:bg-[#5fa000] disabled:opacity-60"
					>
						Create
					</button>
				</div>
			</form>
		</Modal>
	);
};

const NewTagModal = ({
	classification,
	onClose,
	onCreated,
}: {
	classification: string;
	onClose: () => void;
	onCreated: () => void;
}) => {
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [busy, setBusy] = useState(false);
	const [err, setErr] = useState<string | null>(null);

	return (
		<Modal title={`New tag in ${classification}`} onClose={onClose}>
			<form
				onSubmit={async (e) => {
					e.preventDefault();
					setBusy(true);
					setErr(null);
					try {
						await om.tags.create({ name, description, classification });
						onCreated();
					} catch (ex) {
						setErr(ex instanceof Error ? ex.message : String(ex));
					} finally {
						setBusy(false);
					}
				}}
				className="space-y-3"
			>
				<Field label="Name" required>
					<input
						value={name}
						onChange={(e) => setName(e.target.value)}
						required
						pattern="[A-Za-z0-9_\-]+"
						className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
					/>
				</Field>
				<Field label="Description">
					<textarea
						value={description}
						onChange={(e) => setDescription(e.target.value)}
						rows={3}
						className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
					/>
				</Field>
				{err ? <p className="text-xs text-red-600">{err}</p> : null}
				<div className="flex justify-end gap-2 pt-2">
					<button
						type="button"
						onClick={onClose}
						className="rounded-lg px-3 py-1.5 text-sm text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
					>
						Cancel
					</button>
					<button
						type="submit"
						disabled={busy}
						className="rounded-lg bg-[#76b900] px-3 py-1.5 text-sm font-medium text-white hover:bg-[#5fa000] disabled:opacity-60"
					>
						Create
					</button>
				</div>
			</form>
		</Modal>
	);
};

const Field = ({
	label,
	required,
	children,
}: {
	label: string;
	required?: boolean;
	children: React.ReactNode;
}) => (
	<label className="block">
		<span className="mb-1 block text-xs font-medium text-zinc-700 dark:text-zinc-300">
			{label}
			{required ? <span className="text-red-500"> *</span> : null}
		</span>
		{children}
	</label>
);
