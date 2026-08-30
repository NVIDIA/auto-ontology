// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import type { ChangeEvent } from 'react';
import { Spinner } from '@nvidia/foundations-react-core';
import { Button } from '@/common/Button';
import { Size, ButtonTheme } from '@/enums/button';
import {
	CONNECTION_FIELDS,
	connectionDisplayName,
	type ConnectionFieldKey,
	type ConnectionType,
} from '@/enums/connection';

export type ConnectionConnectStepProps = {
	connectionType: ConnectionType;
	values: Partial<Record<ConnectionFieldKey, string>>;
	onFieldChange: (key: ConnectionFieldKey, value: string) => void;
	loading?: boolean;
	testSuccessMessage?: string | null;
	onTestConnection?: () => void;
	testDisabled?: boolean;
	testingConnection?: boolean;
};

export const ConnectionConnectStep = ({
	connectionType,
	values,
	onFieldChange,
	loading = false,
	testSuccessMessage = null,
	onTestConnection,
	testDisabled = false,
	testingConnection = false,
}: ConnectionConnectStepProps) => {
	if (loading) {
		return (
			<div className="flex flex-1 items-center justify-center">
				<Spinner aria-label="Loading connection" className="h-10 w-10" />
			</div>
		);
	}

	const testButtonDisabled = testDisabled || testingConnection;
	const fields = CONNECTION_FIELDS[connectionType];

	return (
		<div className="flex flex-1 flex-col gap-4 p-2">
			<p className="text-sm text-zinc-600 dark:text-zinc-400">
				Connect to {connectionDisplayName[connectionType]}
			</p>
			<div className="flex w-full flex-col gap-3">
				{fields.map((field) => {
					const onChange = ({
						target,
					}: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
						onFieldChange(field.key, target.value);
					const inputClassName =
						'rounded-lg border border-zinc-300 bg-white px-3 py-2 font-mono text-sm text-zinc-900 outline-none focus:border-[#76b900] focus:ring-2 focus:ring-[#76b900]/30 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-100';

					// Boolean fields render as a checkbox; the modal stores every
					// value as a string, so 'true'/'' is the on/off representation.
					if (field.boolean) {
						return (
							<label key={field.key} className="flex flex-col gap-1.5">
								<span className="flex items-center gap-2 text-sm font-medium text-zinc-800 dark:text-zinc-200">
									<input
										type="checkbox"
										checked={values[field.key] === 'true'}
										onChange={({ target }) =>
											onFieldChange(field.key, target.checked ? 'true' : '')
										}
										data-testid={`connection-field-${field.key}`}
										className="h-4 w-4 rounded border-zinc-300 accent-[#76b900] dark:border-zinc-600"
									/>
									{field.label}
								</span>
								{field.hint != null && (
									<span className="text-xs text-zinc-500 dark:text-zinc-400">
										{field.hint}
									</span>
								)}
							</label>
						);
					}

					// File fields carry the picked file's bytes as base64 so the
					// keystore travels with the connection instead of having to
					// exist at a path on the server.
					if (field.file) {
						const onFile = async ({ target }: ChangeEvent<HTMLInputElement>) => {
							const picked = target.files?.[0];
							if (!picked) {
								onFieldChange(field.key, '');
								return;
							}
							const bytes = new Uint8Array(await picked.arrayBuffer());
							let binary = '';
							bytes.forEach((byte) => {
								binary += String.fromCharCode(byte);
							});
							onFieldChange(field.key, window.btoa(binary));
						};
						const encoded = values[field.key] ?? '';
						return (
							<label key={field.key} className="flex flex-col gap-1.5">
								<span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
									{field.label}
									{field.optional ? null : (
										<span className="ml-0.5 text-red-500">*</span>
									)}
								</span>
								<input
									type="file"
									accept={field.fileAccept}
									onChange={onFile}
									data-testid={`connection-field-${field.key}`}
									className={`${inputClassName} file:mr-3 file:rounded-md file:border-0 file:bg-zinc-100 file:px-3 file:py-1 file:text-sm dark:file:bg-zinc-800 dark:file:text-zinc-200`}
								/>
								{encoded !== '' && (
									<span className="text-xs text-zinc-500 dark:text-zinc-400">
										{`Loaded ${Math.round((encoded.length * 3) / 4 / 1024)} KB`}
									</span>
								)}
								{field.hint != null && (
									<span className="text-xs text-zinc-500 dark:text-zinc-400">
										{field.hint}
									</span>
								)}
							</label>
						);
					}

					return (
						<label key={field.key} className="flex flex-col gap-1.5">
							<span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
								{field.label}
								{field.optional ? null : (
									<span className="ml-0.5 text-red-500">*</span>
								)}
							</span>
							{field.multiline ? (
								// A PEM key is multi-line, and its newlines are
								// significant, so it cannot use a text input.
								<textarea
									value={values[field.key] ?? ''}
									onChange={onChange}
									placeholder={field.placeholder}
									rows={5}
									spellCheck={false}
									data-testid={`connection-field-${field.key}`}
									className={`${inputClassName} resize-y`}
								/>
							) : (
								<input
									type={field.secret ? 'password' : 'text'}
									value={values[field.key] ?? ''}
									onChange={onChange}
									placeholder={field.placeholder}
									autoComplete={field.secret ? 'new-password' : 'off'}
									data-testid={`connection-field-${field.key}`}
									className={inputClassName}
								/>
							)}
							{field.hint != null && (
								<span className="text-xs text-zinc-500 dark:text-zinc-400">
									{field.hint}
								</span>
							)}
						</label>
					);
				})}
			</div>
			<div className="mt-auto flex flex-col items-start">
				{onTestConnection != null && (
					<Button
						theme={ButtonTheme.Outline}
						size={Size.REGULAR}
						type="button"
						onClick={onTestConnection}
						disabled={testButtonDisabled}
					>
						{testingConnection ? 'Test Connection…' : 'Test Connection'}
					</Button>
				)}
				{testSuccessMessage != null && (
					<p className="text-sm text-[#5e9400] dark:text-[#8fd100]">
						{testSuccessMessage}
					</p>
				)}
			</div>
		</div>
	);
};
