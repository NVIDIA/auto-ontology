// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Icon, IconName } from '@/common/icons';
import { ModalCreateNewItem, ConfirmModal } from '@/common/modal';
import { PopoverMenu } from '@/common/PopoverMenu';
import { acronymsApi, promptsApi, type Acronym, type Prompt } from '@/api/settings';

type SettingsSectionProps = {
	title: string;
	subtitle: string;
	prompts: Prompt[];
	loading: boolean;
	onCreate: (content: string) => Promise<void>;
	onUpdate: (id: string, content: string) => Promise<void>;
	onDelete: (id: string) => Promise<void>;
};

const PromptEditor = ({
	initialValue,
	onSave,
	onCancel,
}: {
	initialValue: string;
	onSave: (content: string) => void;
	onCancel: () => void;
}) => {
	const [value, setValue] = useState(initialValue);
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	useEffect(() => {
		textareaRef.current?.focus();
	}, []);

	return (
		<div className="space-y-3 p-4">
			<div className="rounded-lg border border-zinc-200/90 bg-white p-4 dark:border-zinc-700/90 dark:bg-zinc-900/60">
				<span className="mb-2 block text-xs font-semibold uppercase tracking-wide text-zinc-500">
					Prompt
				</span>
				<div className="space-y-2">
					<textarea
						ref={textareaRef}
						value={value}
						onChange={(e) => setValue(e.target.value)}
						rows={3}
						placeholder="Enter your custom prompt prefix…"
						className="w-full resize-y rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm leading-relaxed text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-600"
					/>
					<div className="flex justify-end gap-2">
						<button
							type="button"
							onClick={onCancel}
							className="cursor-pointer rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-400 dark:hover:bg-zinc-800"
						>
							Cancel
						</button>
						<button
							type="button"
							onClick={() => onSave(value)}
							className="cursor-pointer rounded-md bg-[#76b900] px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-[#6aa500]"
						>
							Save
						</button>
					</div>
				</div>
			</div>
		</div>
	);
};

const AcronymRow = ({
	acronym,
	onEdit,
	onDelete,
}: {
	acronym: Acronym;
	onEdit: (a: Acronym) => void;
	onDelete: (a: Acronym) => void;
}) => (
	<tr className="border-b border-zinc-100 transition-colors last:border-b-0 hover:bg-zinc-50/80 dark:border-zinc-800 dark:hover:bg-zinc-900/50">
		<td className="px-3 py-2 font-medium text-zinc-800 dark:text-zinc-200">{acronym.name}</td>
		<td className="px-3 py-2 text-zinc-800 dark:text-zinc-200">{acronym.description}</td>
		<td className="relative px-2 py-2 text-right">
			<PopoverMenu
				items={[
					{
						label: 'Edit',
						icon: <Icon name={IconName.Pencil} className="h-3.5 w-3.5" />,
						onClick: () => onEdit(acronym),
					},
					{
						label: 'Delete',
						icon: <Icon name={IconName.Trash} className="h-3.5 w-3.5" />,
						onClick: () => onDelete(acronym),
						danger: true,
					},
				]}
				trigger={({ toggle }) => (
					<button
						type="button"
						onClick={toggle}
						className="cursor-pointer rounded-md p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-600 dark:hover:bg-zinc-700 dark:hover:text-zinc-300"
					>
						<Icon name={IconName.DotsVertical} className="h-4 w-4" />
					</button>
				)}
			/>
		</td>
	</tr>
);

const AcronymsSection = () => {
	const [acronyms, setAcronyms] = useState<Acronym[]>([]);
	const [modalOpen, setModalOpen] = useState(false);
	const [editingAcronym, setEditingAcronym] = useState<Acronym | null>(null);
	const [confirmDelete, setConfirmDelete] = useState<Acronym | null>(null);
	const [name, setName] = useState('');
	const [description, setDescription] = useState('');
	const [nameExists, setNameExists] = useState(false);
	const [checkingName, setCheckingName] = useState(false);
	const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	const isEditing = editingAcronym !== null;

	const fetchAcronyms = useCallback(async () => {
		const data = await acronymsApi.get();
		setAcronyms(data);
	}, []);

	useEffect(() => {
		fetchAcronyms();
	}, [fetchAcronyms]);

	useEffect(() => {
		const trimmed = name.trim();
		if (!trimmed || (isEditing && trimmed === editingAcronym.name)) {
			setNameExists(false);
			setCheckingName(false);
			return;
		}

		setCheckingName(true);
		if (debounceRef.current) clearTimeout(debounceRef.current);

		debounceRef.current = setTimeout(async () => {
			try {
				const { exists } = await acronymsApi.checkName(trimmed);
				setNameExists(exists);
			} catch {
				setNameExists(false);
			} finally {
				setCheckingName(false);
			}
		}, 1000);

		return () => {
			if (debounceRef.current) clearTimeout(debounceRef.current);
		};
	}, [name, isEditing, editingAcronym]);

	const openAddModal = () => {
		setEditingAcronym(null);
		setName('');
		setDescription('');
		setModalOpen(true);
	};

	const openEditModal = (a: Acronym) => {
		setEditingAcronym(a);
		setName(a.name);
		setDescription(a.description);
		setModalOpen(true);
	};

	const handleClose = () => {
		setModalOpen(false);
		setEditingAcronym(null);
		setName('');
		setDescription('');
		setNameExists(false);
		setCheckingName(false);
		if (debounceRef.current) clearTimeout(debounceRef.current);
	};

	const canSubmit = name.trim() && description.trim() && !nameExists && !checkingName;

	const handleSubmit = async () => {
		if (!canSubmit) return;
		if (isEditing) {
			await acronymsApi.update(editingAcronym.id, {
				name: name.trim(),
				description: description.trim(),
			});
		} else {
			await acronymsApi.create({ name: name.trim(), description: description.trim() });
		}
		handleClose();
		await fetchAcronyms();
	};

	const handleDelete = async () => {
		if (!confirmDelete) return;
		await acronymsApi.delete(confirmDelete.id);
		setConfirmDelete(null);
		await fetchAcronyms();
	};

	const hasAcronyms = acronyms.length > 0;

	return (
		<div className="rounded-lg border border-zinc-200/90 bg-white/90 p-5 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
			<div className="flex items-center justify-between">
				<div className="flex items-baseline gap-2">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						Glossary
					</h2>
					<span className="text-sm text-zinc-500 dark:text-zinc-400">
						Define definitions for your account, department, or database context
					</span>
				</div>
				{hasAcronyms && (
					<button
						type="button"
						onClick={openAddModal}
						className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-[#76b900]/40 px-3 py-1.5 text-sm font-medium text-[#76b900] transition-colors hover:bg-[#76b900]/10"
					>
						<Icon name={IconName.ChatBubble} className="h-3.5 w-3.5" />
						Add
					</button>
				)}
			</div>

			{hasAcronyms ? (
				<div className="mt-4 overflow-visible rounded-md border border-zinc-200/90 dark:border-zinc-700">
					<table className="w-full min-w-[28rem] text-left text-sm">
						<thead className="border-b border-zinc-200 bg-zinc-100/95 dark:border-zinc-700 dark:bg-zinc-800/90">
							<tr>
								<th className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-600 dark:text-zinc-300">
									Name
								</th>
								<th className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-600 dark:text-zinc-300">
									Description
								</th>
								<th className="w-10" />
							</tr>
						</thead>
						<tbody>
							{acronyms.map((a) => (
								<AcronymRow
									key={a.id}
									acronym={a}
									onEdit={openEditModal}
									onDelete={setConfirmDelete}
								/>
							))}
						</tbody>
					</table>
				</div>
			) : (
				<div className="mt-4 flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-zinc-300/90 bg-white/70 px-8 py-10 dark:border-zinc-600 dark:bg-zinc-900/30">
					<p className="text-sm text-zinc-400 dark:text-zinc-500">
						No Glossary definitions created yet
					</p>
					<button
						type="button"
						onClick={openAddModal}
						className="flex cursor-pointer items-center gap-2 rounded-lg bg-[#76b900] px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[#6aa500]"
					>
						<Icon name={IconName.ChatBubble} className="h-4 w-4" />
						Add Definition
					</button>
				</div>
			)}

			<ModalCreateNewItem
				open={modalOpen}
				onClose={handleClose}
				title={isEditing ? 'Edit Glossary Definition' : 'Add Glossary Definition'}
				submitLabel="Save"
				onSubmit={handleSubmit}
				canSubmit={Boolean(canSubmit)}
				accent="teal"
			>
				<div>
					<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
						Name
					</label>
					<input
						type="text"
						value={name}
						onChange={(e) => setName(e.target.value)}
						placeholder="Name"
						className={`w-full rounded-lg border bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500 ${nameExists ? 'border-red-400 focus:border-red-500 focus:ring-2 focus:ring-red-500/30 dark:border-red-500 dark:focus:border-red-400 dark:focus:ring-red-400/30' : 'border-zinc-300 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/30 dark:border-zinc-600 dark:focus:border-teal-400 dark:focus:ring-teal-400/30'}`}
					/>
					{nameExists && (
						<p className="mt-1 text-xs text-red-500 dark:text-red-400">
							An acronym with this name already exists
						</p>
					)}
					{checkingName && name.trim() && (
						<p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
							Checking name…
						</p>
					)}
				</div>
				<div>
					<label className="mb-1.5 block text-sm font-medium text-zinc-900 dark:text-zinc-100">
						Description
					</label>
					<textarea
						value={description}
						onChange={(e) => setDescription(e.target.value)}
						placeholder="Add Short Description"
						rows={4}
						className="w-full resize-y rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-700 outline-none transition-colors placeholder:text-zinc-400 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-300 dark:placeholder:text-zinc-500 dark:focus:border-teal-400 dark:focus:ring-teal-400/30"
					/>
				</div>
			</ModalCreateNewItem>

			<ConfirmModal
				open={confirmDelete !== null}
				title="Delete acronym"
				message="Are you sure you want to delete this acronym? This action cannot be undone."
				onConfirm={handleDelete}
				onCancel={() => setConfirmDelete(null)}
			/>
		</div>
	);
};

const SettingsSection = ({
	title,
	subtitle,
	prompts,
	loading,
	onCreate,
	onUpdate,
	onDelete,
}: SettingsSectionProps) => {
	const hasPrompts = prompts.length > 0;
	const [editing, setEditing] = useState(false);

	const handleButtonClick = () => {
		setEditing(true);
	};

	const handleSave = async (content: string) => {
		const trimmed = content.trim();
		if (hasPrompts) {
			if (trimmed) {
				await onUpdate(prompts[0].id, trimmed);
			} else {
				await onDelete(prompts[0].id);
			}
		} else if (trimmed) {
			await onCreate(trimmed);
		}
		setEditing(false);
	};

	const handleCancel = () => {
		setEditing(false);
	};

	return (
		<div className="rounded-lg border border-zinc-200/90 bg-white/90 shadow-sm ring-1 ring-zinc-950/[0.04] dark:border-zinc-700/90 dark:bg-zinc-950/50 dark:ring-white/[0.06]">
			<div className="flex items-center justify-between border-b border-zinc-200/90 px-5 py-3.5 dark:border-zinc-700/90">
				<div className="flex items-baseline gap-2">
					<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
						{title}
					</h2>
					<span className="text-sm text-zinc-500 dark:text-zinc-400">{subtitle}</span>
				</div>
				<button
					type="button"
					onClick={handleButtonClick}
					className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-[#76b900]/40 px-3 py-1.5 text-sm font-medium text-[#76b900] transition-colors hover:bg-[#76b900]/10"
				>
					<Icon name={IconName.Pencil} className="h-3.5 w-3.5" />
					{hasPrompts ? 'Edit' : 'Add'}
				</button>
			</div>

			{editing ? (
				<PromptEditor
					initialValue={hasPrompts ? prompts[0].content : ''}
					onSave={handleSave}
					onCancel={handleCancel}
				/>
			) : hasPrompts ? (
				<div className="space-y-3 p-4">
					<div className="rounded-lg border border-zinc-200/90 bg-white p-4 dark:border-zinc-700/90 dark:bg-zinc-900/60">
						<span className="mb-2 block text-xs font-semibold uppercase tracking-wide text-zinc-500">
							Prompt
						</span>
						<p className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
							{prompts[0].content}
						</p>
					</div>
				</div>
			) : !loading ? (
				<div className="flex flex-col items-center justify-center gap-2 rounded-b-lg bg-zinc-50/80 px-8 py-10 dark:bg-zinc-900/30">
					<Icon
						name={IconName.ChatBubble}
						className="h-8 w-8 text-zinc-300 dark:text-zinc-600"
					/>
					<p className="text-sm text-zinc-400 dark:text-zinc-500">No Prompt</p>
				</div>
			) : null}
		</div>
	);
};

export default function SemanticInputSettingsPage() {
	const [prompts, setPrompts] = useState<Prompt[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const fetchPrompts = useCallback(async () => {
		try {
			setError(null);
			const data = await promptsApi.get();
			setPrompts(data);
		} catch {
			setError('Failed to load custom prompts.');
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchPrompts();
	}, [fetchPrompts]);

	const handleCreate = async (content: string) => {
		try {
			setError(null);
			const created = await promptsApi.create({ content });
			setPrompts((prev) => [created, ...prev]);
		} catch {
			setError('Failed to create custom prompt.');
		}
	};

	const handleUpdate = async (id: string, content: string) => {
		try {
			setError(null);
			const updated = await promptsApi.update(id, { content });
			setPrompts((prev) => prev.map((p) => (p.id === id ? updated : p)));
		} catch {
			setError('Failed to save custom prompt.');
		}
	};

	const handleDelete = async (id: string) => {
		try {
			setError(null);
			await promptsApi.delete(id);
			setPrompts((prev) => prev.filter((p) => p.id !== id));
		} catch {
			setError('Failed to delete custom prompt.');
		}
	};

	return (
		<>
			{error ? (
				<div className="shrink-0 border-b border-red-200/90 bg-red-50 px-7 py-3 text-sm text-red-800 sm:px-10 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
					{error}
				</div>
			) : null}

			<main className="min-h-0 min-w-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,rgba(255,255,255,1)_0%,rgba(250,250,250,0.6)_100%)] px-7 py-6 sm:px-10 sm:py-7 dark:bg-[linear-gradient(180deg,rgba(9,9,11,1)_0%,rgba(24,24,27,0.5)_100%)]">
				<div className="w-full space-y-5">
					<SettingsSection
						title="Custom prompts"
						subtitle="Default prompt prefixes automatically applied to all user prompts in this account"
						prompts={prompts}
						loading={loading}
						onCreate={handleCreate}
						onUpdate={handleUpdate}
						onDelete={handleDelete}
					/>
					<AcronymsSection />
				</div>
			</main>
		</>
	);
}
