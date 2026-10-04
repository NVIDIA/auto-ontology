// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import { Button } from '@/common/Button';
import { Icon, IconName } from '@/common/icons';
import { ButtonTheme, Size } from '@/enums/button';

export type FileUploadProps = {
	/** Currently selected file, or `null` when the zone is empty. */
	value: File | null;
	/** Called with the chosen file (or `null` when cleared). Not called when `validate` rejects the file. */
	onChange: (file: File | null) => void;
	/** Forwarded to the underlying `<input type="file" accept="...">`. */
	accept?: string;
	disabled?: boolean;
	/** Leading label, e.g. "Click to upload". */
	label?: string;
	/** Caption under the label, e.g. "YAML files (.yaml, .yml)". */
	description?: string;
	/** Return an error message to reject a file (e.g. wrong extension), or `null` to accept it. */
	validate?: (file: File) => string | null;
	/** Called with the message returned by `validate` when a file is rejected. */
	onError?: (message: string) => void;
	'aria-label'?: string;
	className?: string;
};

export const FileUpload = ({
	value,
	onChange,
	accept,
	disabled = false,
	label = 'Click to upload',
	description,
	validate,
	onError,
	'aria-label': ariaLabel = 'Upload a file',
	className = '',
}: FileUploadProps) => {
	const [isDragOver, setIsDragOver] = useState(false);
	const fileInputRef = useRef<HTMLInputElement>(null);

	const applyFile = (file: File | null) => {
		if (disabled) return;
		if (file && validate) {
			const message = validate(file);
			if (message) {
				onError?.(message);
				return;
			}
		}
		onChange(file);
	};

	const handleBrowseClick = () => {
		if (!disabled) fileInputRef.current?.click();
	};

	const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
		applyFile(event.target.files?.[0] ?? null);
	};

	const handleClear = () => {
		applyFile(null);
		if (fileInputRef.current) fileInputRef.current.value = '';
	};

	const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
		event.preventDefault();
		if (!disabled) setIsDragOver(true);
	};

	const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
		event.preventDefault();
		setIsDragOver(false);
	};

	const handleDrop = (event: DragEvent<HTMLDivElement>) => {
		event.preventDefault();
		setIsDragOver(false);
		applyFile(event.dataTransfer.files?.[0] ?? null);
	};

	return (
		<div
			role="button"
			tabIndex={disabled ? -1 : 0}
			onClick={handleBrowseClick}
			onKeyDown={(event) => {
				if (event.key !== 'Enter' && event.key !== ' ') return;
				event.preventDefault();
				handleBrowseClick();
			}}
			onDragOver={handleDragOver}
			onDragLeave={handleDragLeave}
			onDrop={handleDrop}
			aria-disabled={disabled || undefined}
			aria-label={ariaLabel}
			className={`flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors focus-visible:ring-2 focus-visible:ring-[#76b900]/40 focus-visible:outline-none ${
				disabled
					? 'cursor-default border-zinc-200 opacity-60 dark:border-zinc-800'
					: `cursor-pointer ${
							isDragOver
								? 'border-[#76b900] bg-[#76b900]/5'
								: 'border-zinc-300 hover:border-zinc-400 dark:border-zinc-600 dark:hover:border-zinc-500'
						}`
			} ${className}`}
		>
			<Icon name={IconName.Upload} className="h-6 w-6 text-secondary dark:text-zinc-500" />
			{value ? (
				<div className="flex items-center gap-2">
					<span className="text-sm font-medium text-heading dark:text-zinc-200">
						{value.name}
					</span>
					<Button
						theme={ButtonTheme.IconNeutral}
						size={Size.SMALL}
						iconOnly
						type="button"
						disabled={disabled}
						onClick={(event) => {
							event.stopPropagation();
							handleClear();
						}}
						aria-label="Remove selected file"
					>
						<Icon name={IconName.Close} className="h-3.5 w-3.5" />
					</Button>
				</div>
			) : (
				<>
					<p className="text-sm text-body dark:text-zinc-300">
						<span className="font-medium text-[#5e9400] dark:text-[#a3d63a]">
							{label}
						</span>{' '}
						or drag and drop
					</p>
					{description ? <p className="text-xs text-secondary">{description}</p> : null}
				</>
			)}
			<input
				ref={fileInputRef}
				type="file"
				accept={accept}
				disabled={disabled}
				onChange={handleInputChange}
				className="hidden"
			/>
		</div>
	);
};
