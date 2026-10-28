// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useRef, useState, type DragEvent } from 'react';

import { documentsApi } from '@/api/documents';
import { Button } from '@/common/Button';
import { Icon, IconName } from '@/common/icons';
import { ModalCreateNewItem } from '@/common/modal';
import { Spinner } from '@/common/Spinner';
import {
	MAX_DOCUMENT_FILES_PER_UPLOAD,
	MAX_DOCUMENT_UPLOAD_BYTES,
	MAX_DOCUMENT_WORDS,
	SUPPORTED_DOCUMENT_EXTENSIONS,
} from '@/constants/documents';
import { ButtonTheme, Size } from '@/enums/button';

const MAX_UPLOAD_MB = MAX_DOCUMENT_UPLOAD_BYTES / (1024 * 1024);
const ACCEPT = SUPPORTED_DOCUMENT_EXTENSIONS.map((extension) => `.${extension}`).join(',');

type UploadState = 'pending' | 'uploading' | 'done' | 'failed';

type QueuedFile = {
	key: string;
	file: File;
	state: UploadState;
	error?: string;
};

/**
 * Checked here so a wrong file is refused before it is sent. The upload route
 * applies the same two rules and is the authority.
 */
const validateFile = (file: File): string | null => {
	const extension = file.name.split('.').pop()?.toLowerCase() ?? '';
	if (!(SUPPORTED_DOCUMENT_EXTENSIONS as readonly string[]).includes(extension)) {
		return `${file.name}: unsupported file type.`;
	}
	if (file.size > MAX_DOCUMENT_UPLOAD_BYTES) {
		return `${file.name}: exceeds the ${MAX_UPLOAD_MB}MB limit.`;
	}
	return null;
};

const formatSize = (bytes: number): string =>
	bytes >= 1024 * 1024
		? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
		: `${Math.max(1, Math.round(bytes / 1024))} KB`;

const fileKey = (file: File): string => `${file.name}:${file.size}:${file.lastModified}`;

type DocumentUploadModalProps = {
	open: boolean;
	onClose: () => void;
	/** At least one file was stored; each is ingested in the background from here. */
	onUploaded: () => void;
};

export const DocumentUploadModal = ({ open, onClose, onUploaded }: DocumentUploadModalProps) => {
	const [queue, setQueue] = useState<QueuedFile[]>([]);
	const [uploading, setUploading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [isDragOver, setIsDragOver] = useState(false);
	const inputRef = useRef<HTMLInputElement>(null);

	const reset = () => {
		setQueue([]);
		setError(null);
	};

	const handleClose = () => {
		if (uploading) return;
		reset();
		onClose();
	};

	const addFiles = (incoming: File[]) => {
		if (uploading || incoming.length === 0) return;
		const rejected: string[] = [];
		const held = new Set(queue.map((item) => item.key));
		const accepted: QueuedFile[] = [];
		for (const file of incoming) {
			const message = validateFile(file);
			if (message) {
				rejected.push(message);
				continue;
			}
			const key = fileKey(file);
			if (held.has(key)) continue;
			held.add(key);
			accepted.push({ key, file, state: 'pending' });
		}
		const room = MAX_DOCUMENT_FILES_PER_UPLOAD - queue.length;
		if (accepted.length > room) {
			rejected.push(
				`Up to ${MAX_DOCUMENT_FILES_PER_UPLOAD} files at a time; ${accepted.length - Math.max(room, 0)} not added.`,
			);
		}
		setQueue((prev) => [...prev, ...accepted.slice(0, Math.max(room, 0))]);
		setError(rejected.length > 0 ? rejected.join(' ') : null);
	};

	const removeFile = (key: string) => {
		setQueue((prev) => prev.filter((item) => item.key !== key));
		setError(null);
	};

	const setState = (key: string, state: UploadState, message?: string) =>
		setQueue((prev) =>
			prev.map((item) => (item.key === key ? { ...item, state, error: message } : item)),
		);

	// One request per file, in turn: a bad file fails alone instead of taking
	// the batch with it, and the proxy never buffers more than one file.
	const handleUpload = async () => {
		const pending = queue.filter((item) => item.state === 'pending' || item.state === 'failed');
		if (pending.length === 0 || uploading) return;
		setUploading(true);
		setError(null);
		let stored = 0;
		let failed = 0;
		for (const item of pending) {
			setState(item.key, 'uploading');
			const response = await documentsApi.upload(item.file);
			if (response.error) {
				failed += 1;
				setState(item.key, 'failed', response.message ?? 'Failed to upload file.');
			} else {
				stored += 1;
				setState(item.key, 'done');
			}
		}
		setUploading(false);
		if (stored > 0) onUploaded();
		if (failed === 0) {
			reset();
			onClose();
		} else {
			setQueue((prev) => prev.filter((item) => item.state !== 'done'));
		}
	};

	const handleDrop = (event: DragEvent<HTMLDivElement>) => {
		event.preventDefault();
		setIsDragOver(false);
		addFiles(Array.from(event.dataTransfer.files ?? []));
	};

	const sendable = queue.filter((item) => item.state !== 'done').length;
	const retrying = queue.some((item) => item.state === 'failed');

	return (
		<ModalCreateNewItem
			open={open}
			onClose={handleClose}
			title="File Upload"
			submitLabel={
				uploading
					? 'Uploading…'
					: retrying
						? 'Retry'
						: `Upload${sendable > 1 ? ` ${sendable} Files` : ''}`
			}
			canSubmit={sendable > 0 && !uploading}
			onSubmit={handleUpload}
			className="w-[560px] max-w-full"
		>
			<div
				role="button"
				tabIndex={uploading ? -1 : 0}
				onClick={() => !uploading && inputRef.current?.click()}
				onKeyDown={(event) => {
					if (event.key !== 'Enter' && event.key !== ' ') return;
					event.preventDefault();
					if (!uploading) inputRef.current?.click();
				}}
				onDragOver={(event) => {
					event.preventDefault();
					if (!uploading) setIsDragOver(true);
				}}
				onDragLeave={(event) => {
					event.preventDefault();
					setIsDragOver(false);
				}}
				onDrop={handleDrop}
				aria-disabled={uploading || undefined}
				aria-label="Upload documents"
				className={`flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors focus-visible:ring-2 focus-visible:ring-[#76b900]/40 focus-visible:outline-none ${
					uploading
						? 'cursor-default border-zinc-200 opacity-60 dark:border-zinc-800'
						: `cursor-pointer ${
								isDragOver
									? 'border-[#76b900] bg-[#76b900]/5'
									: 'border-zinc-300 hover:border-zinc-400 dark:border-zinc-600 dark:hover:border-zinc-500'
							}`
				}`}
			>
				<Icon
					name={IconName.Upload}
					className="h-6 w-6 text-secondary dark:text-zinc-500"
				/>
				<p className="text-sm text-body dark:text-zinc-300">
					Drag your files here or{' '}
					<span className="font-medium text-[#5e9400] dark:text-[#a3d63a]">
						upload from your computer
					</span>
				</p>
				<p className="text-xs text-secondary">
					Up to {MAX_DOCUMENT_FILES_PER_UPLOAD} files · {MAX_UPLOAD_MB}MB or{' '}
					{MAX_DOCUMENT_WORDS.toLocaleString()} words per file
				</p>
				<input
					ref={inputRef}
					type="file"
					multiple
					accept={ACCEPT}
					disabled={uploading}
					onChange={(event) => {
						addFiles(Array.from(event.target.files ?? []));
						event.target.value = '';
					}}
					className="hidden"
				/>
			</div>

			<p className="text-xs text-secondary dark:text-zinc-500">
				Supported file types: {SUPPORTED_DOCUMENT_EXTENSIONS.join(', ')}. A name that is
				already taken gets a number added, e.g. report (1).pdf.
			</p>

			{queue.length > 0 ? (
				<ul className="max-h-60 divide-y divide-zinc-200 overflow-y-auto rounded-lg border border-zinc-200 dark:divide-zinc-700 dark:border-zinc-700">
					{queue.map((item) => (
						<li key={item.key} className="flex items-center gap-3 px-3 py-2 text-sm">
							<div className="min-w-0 flex-1">
								<p className="truncate text-heading dark:text-zinc-200">
									{item.file.name}
								</p>
								<p
									className={`truncate text-xs ${item.state === 'failed' ? 'text-red-700 dark:text-red-300' : 'text-secondary dark:text-zinc-500'}`}
								>
									{item.state === 'failed'
										? item.error
										: formatSize(item.file.size)}
								</p>
							</div>
							{item.state === 'uploading' ? (
								<Spinner className="h-4 w-4 text-secondary" />
							) : item.state === 'done' ? (
								<Icon name={IconName.Check} className="h-4 w-4 text-[#76b900]" />
							) : (
								<Button
									theme={ButtonTheme.IconNeutral}
									size={Size.SMALL}
									iconOnly
									type="button"
									disabled={uploading}
									onClick={() => removeFile(item.key)}
									aria-label={`Remove ${item.file.name}`}
								>
									<Icon name={IconName.Close} className="h-3.5 w-3.5" />
								</Button>
							)}
						</li>
					))}
				</ul>
			) : null}

			{error ? (
				<div className="rounded-lg border border-red-200/90 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
					{error}
				</div>
			) : null}
		</ModalCreateNewItem>
	);
};
