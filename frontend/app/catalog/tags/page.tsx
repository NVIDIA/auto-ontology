// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
	Badge,
	Button,
	FormField,
	Modal,
	ModalCloseButton,
	Skeleton,
	Text,
	TextArea,
	TextInput,
} from '@kui/foundations-react';
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
					<div className="flex items-center gap-2">
						<Text asChild kind="title/md">
							<h1 className="text-[var(--text-color-primary)]">Tags</h1>
						</Text>
						{data != null ? (
							<Badge color="gray" kind="outline">
								{totalTags}
							</Badge>
						) : null}
					</div>
					<p className="mt-1 max-w-2xl text-sm text-[var(--text-color-base)]">
						Classifications group related tags. Use them to label tables and columns
						with custom themes (e.g. <span className="font-medium">Sensitivity</span>)
						or rely on the built-in <span className="font-medium">PII</span> system
						classification populated by Auto-Classification.
					</p>
				</div>
				<Button color="brand" onClick={() => setNewClassOpen(true)}>
					New classification
				</Button>
			</header>

			{err ? (
				<div className="mb-4 rounded-[var(--radius-lg)] border border-[var(--border-color-feedback-danger)] bg-[var(--background-color-feedback-danger-subtle-hover)] p-4 text-sm text-[var(--text-color-feedback-danger-strong)]">
					{err}
				</div>
			) : null}

			<div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
				<Panel title="Classifications" className="self-start">
					{data == null ? (
						<SkeletonList />
					) : (
						<ul className="flex flex-col gap-1">
							{data.map((c) => (
								<li key={c.name}>
									<button
										type="button"
										onClick={() => setSelected(c.name)}
										className={`flex w-full items-center justify-between rounded-[var(--radius-md)] px-3 py-2 text-left text-sm transition-colors ${
											c.name === selected
												? 'bg-[var(--background-color-accent-green-subtle)] text-[var(--text-color-brand)]'
												: 'text-[var(--text-color-secondary)] hover:bg-[var(--background-color-interaction-hover)]'
										}`}
									>
										<span className="flex items-center gap-2">
											<span className="font-medium">{c.name}</span>
											{c.provider === 'system' ? (
												<Badge color="gray" kind="outline">
													system
												</Badge>
											) : null}
										</span>
										<span className="text-xs tabular-nums text-[var(--text-color-subtle)]">
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
						<p className="text-sm text-[var(--text-color-subtle)]">
							Select a classification…
						</p>
					</Panel>
				) : (
					<Panel
						title={current.name}
						action={
							current.provider === 'system' ? (
								<Badge color="gray" kind="outline">
									system
								</Badge>
							) : (
								<Button
									kind="secondary"
									size="small"
									onClick={() => setCreatingFor(current.name)}
								>
									Add tag
								</Button>
							)
						}
					>
						{current.description ? (
							<p className="mb-4 max-w-3xl border-b border-[var(--border-color-base)] pb-4 text-sm text-[var(--text-color-base)]">
								{current.description}
							</p>
						) : null}

						{current.tags.length === 0 ? (
							<p className="rounded-[var(--radius-lg)] border border-dashed border-[var(--border-color-base)] p-8 text-center text-sm text-[var(--text-color-subtle)]">
								No tags in this classification yet.
							</p>
						) : (
							<ul className="space-y-2">
								{current.tags.map((t) => (
									<li
										key={t.fullyQualifiedName}
										className="flex items-start justify-between gap-4 rounded-[var(--radius-md)] border border-[var(--border-color-base)] p-3"
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
													<Badge color="purple" kind="outline">
														exclusive
													</Badge>
												) : null}
											</div>
											{t.description ? (
												<p className="mt-1.5 text-xs text-[var(--text-color-base)]">
													{t.description}
												</p>
											) : null}
										</div>
										<code className="shrink-0 text-[10px] text-[var(--text-color-subtle)]">
											{t.fullyQualifiedName}
										</code>
									</li>
								))}
							</ul>
						)}
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
	<ul className="flex flex-col gap-2">
		{Array.from({ length: 6 }).map((_, i) => (
			<li key={i}>
				<Skeleton kind="line" />
			</li>
		))}
	</ul>
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

	const submit = async (e: React.FormEvent) => {
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
	};

	return (
		<Modal
			open
			onOpenChange={(o) => {
				if (!o) onClose();
			}}
			slotHeading="New classification"
			renderContent={({ children }) => <form onSubmit={submit}>{children}</form>}
			slotFooter={
				<>
					<ModalCloseButton kind="tertiary">Cancel</ModalCloseButton>
					<Button type="submit" color="brand" disabled={busy}>
						Create
					</Button>
				</>
			}
		>
			<div className="space-y-3">
				<FormField slotLabel="Name" required>
					<TextInput
						value={name}
						onValueChange={(v) => setName(v)}
						attributes={{
							Input: {
								required: true,
								pattern: '[A-Za-z0-9_\\-]+',
								title: 'Letters, digits, underscore, hyphen',
							},
						}}
					/>
				</FormField>
				<FormField slotLabel="Description">
					<TextArea
						value={description}
						onValueChange={(v) => setDescription(v)}
						placeholder="What does this classification group?"
					/>
				</FormField>
				{err ? (
					<p className="text-xs text-[var(--text-color-feedback-danger)]">{err}</p>
				) : null}
			</div>
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

	const submit = async (e: React.FormEvent) => {
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
	};

	return (
		<Modal
			open
			onOpenChange={(o) => {
				if (!o) onClose();
			}}
			slotHeading={`New tag in ${classification}`}
			renderContent={({ children }) => <form onSubmit={submit}>{children}</form>}
			slotFooter={
				<>
					<ModalCloseButton kind="tertiary">Cancel</ModalCloseButton>
					<Button type="submit" color="brand" disabled={busy}>
						Create
					</Button>
				</>
			}
		>
			<div className="space-y-3">
				<FormField slotLabel="Name" required>
					<TextInput
						value={name}
						onValueChange={(v) => setName(v)}
						attributes={{ Input: { required: true, pattern: '[A-Za-z0-9_\\-]+' } }}
					/>
				</FormField>
				<FormField slotLabel="Description">
					<TextArea
						value={description}
						onValueChange={(v) => setDescription(v)}
						placeholder="What does this tag mean?"
					/>
				</FormField>
				{err ? (
					<p className="text-xs text-[var(--text-color-feedback-danger)]">{err}</p>
				) : null}
			</div>
		</Modal>
	);
};
