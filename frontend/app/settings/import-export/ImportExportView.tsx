// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useEffect, useState } from 'react';
import { datasources } from '@/api/datasources';
import { modelInterchangeApi } from '@/api/modelInterchange';
import { Button } from '@/common/Button';
import { FileUpload } from '@/common/FileUpload';
import { Icon, IconName } from '@/common/icons';
import { ConfirmModal } from '@/common/modal';
import { SkeletonRows } from '@/common/Skeleton';
import { Toast } from '@/common/Toast';
import { Toggle } from '@/common/Toggle';
import { ButtonTheme, Size } from '@/enums/button';
import type { Database } from '@/types/datasources';
import type { ImportEntityCounts, ImportSummary } from '@/types/modelInterchange';

const EXPORT_FILENAME = 'gsf-model.yaml';

const downloadBlob = (blob: Blob, filename: string) => {
	const url = URL.createObjectURL(blob);
	const link = document.createElement('a');
	link.href = url;
	link.download = filename;
	document.body.appendChild(link);
	link.click();
	link.remove();
	URL.revokeObjectURL(url);
};

const createdCountEntries = (counts: ImportEntityCounts): [string, number][] =>
	Object.entries(counts).filter(([, value]) => value > 0);

const checkboxClassName = 'h-4 w-4 rounded border-zinc-300 text-[#76b900] focus:ring-[#76b900]/30';

const isYamlFile = (file: File): boolean => /\.ya?ml$/i.test(file.name);

const validateYamlFile = (file: File): string | null =>
	isYamlFile(file) ? null : 'Please choose a .yaml or .yml file.';

export const ImportExportView = () => {
	const [databases, setDatabases] = useState<Database[]>([]);
	const [databasesLoading, setDatabasesLoading] = useState(true);
	const [databasesError, setDatabasesError] = useState<string | null>(null);
	const [selectedDbIds, setSelectedDbIds] = useState<Set<string>>(new Set());

	const [exporting, setExporting] = useState(false);
	const [exportError, setExportError] = useState<string | null>(null);
	const [exportMessage, setExportMessage] = useState<string | null>(null);

	const [importFile, setImportFile] = useState<File | null>(null);
	const [replace, setReplace] = useState(true);
	const [embed, setEmbed] = useState(true);
	const [confirmImportOpen, setConfirmImportOpen] = useState(false);
	const [importing, setImporting] = useState(false);
	const [importError, setImportError] = useState<string | null>(null);
	const [importSummary, setImportSummary] = useState<ImportSummary | null>(null);

	useEffect(() => {
		let active = true;
		void (async () => {
			setDatabasesLoading(true);
			const res = await datasources.getDBs();
			if (!active) return;
			if (res.error) {
				setDatabasesError(res.message ?? 'Failed to load databases.');
				setDatabases([]);
			} else {
				const data = res.data ?? [];
				setDatabasesError(null);
				setDatabases(data);
				// Default to exporting everything.
				setSelectedDbIds(new Set(data.map((db) => db.id)));
			}
			setDatabasesLoading(false);
		})();
		return () => {
			active = false;
		};
	}, []);

	const allSelected = databases.length > 0 && selectedDbIds.size === databases.length;

	const toggleDatabase = (id: string) => {
		setSelectedDbIds((prev) => {
			const next = new Set(prev);
			if (next.has(id)) {
				next.delete(id);
			} else {
				next.add(id);
			}
			return next;
		});
	};

	const toggleSelectAll = () => {
		setSelectedDbIds(allSelected ? new Set() : new Set(databases.map((db) => db.id)));
	};

	const handleExport = async () => {
		if (selectedDbIds.size === 0 || exporting) return;
		setExporting(true);
		setExportError(null);
		const res = await modelInterchangeApi.exportModel(Array.from(selectedDbIds));
		setExporting(false);

		if (res.error) {
			setExportError(res.message ?? 'Failed to export model.');
			return;
		}

		downloadBlob(res.blob, EXPORT_FILENAME);
		setExportMessage('Model exported.');
	};

	const handleImportFileChange = (file: File | null) => {
		setImportFile(file);
		setImportError(null);
		setImportSummary(null);
	};

	const handleRequestImport = () => {
		if (!importFile || importing) return;
		setImportError(null);
		setConfirmImportOpen(true);
	};

	const handleCancelImport = () => {
		if (importing) return;
		setConfirmImportOpen(false);
	};

	const handleConfirmImport = async () => {
		if (!importFile) return;
		setImporting(true);
		setImportError(null);
		const res = await modelInterchangeApi.importModel(importFile, { replace, embed });
		setImporting(false);

		if (res.error) {
			setImportError(res.message ?? 'Failed to import model.');
			return;
		}

		setImportSummary(res.summary);
		setConfirmImportOpen(false);
		setImportFile(null);
	};

	return (
		<div className="mx-auto w-full max-w-2xl space-y-6">
			<h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
				Import / Export
			</h1>

			<section className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-700">
				<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
					Export model
				</h2>
				<p className="mt-1 text-xs text-zinc-500">
					Download the catalog and semantic layer for the selected databases as a YAML
					file.
				</p>

				{databasesLoading ? (
					<div className="mt-4">
						<SkeletonRows rows={3} />
					</div>
				) : databasesError ? (
					<p className="mt-4 text-sm text-red-600 dark:text-red-400">{databasesError}</p>
				) : databases.length === 0 ? (
					<p className="mt-4 text-sm text-zinc-500">No databases found.</p>
				) : (
					<div className="mt-4 space-y-2">
						<label className="flex cursor-pointer items-center gap-3 rounded-md px-2 py-1.5 hover:bg-zinc-50 dark:hover:bg-zinc-800/80">
							<input
								type="checkbox"
								checked={allSelected}
								onChange={toggleSelectAll}
								className={checkboxClassName}
							/>
							<span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
								{allSelected ? 'Deselect all' : 'Select all'}
							</span>
						</label>
						<div className="max-h-56 overflow-y-auto rounded-md border border-zinc-100 dark:border-zinc-800">
							{databases.map((db) => (
								<label
									key={db.id}
									className="flex cursor-pointer items-center gap-3 border-b border-zinc-100 px-3 py-2 last:border-b-0 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-800/80"
								>
									<input
										type="checkbox"
										checked={selectedDbIds.has(db.id)}
										onChange={() => toggleDatabase(db.id)}
										className={checkboxClassName}
									/>
									<span className="text-sm text-zinc-800 dark:text-zinc-200">
										{db.name}
									</span>
								</label>
							))}
						</div>
					</div>
				)}

				{exportError ? (
					<p className="mt-3 text-sm text-red-600 dark:text-red-400">{exportError}</p>
				) : null}

				<div className="mt-4 flex justify-end">
					<Button
						theme={ButtonTheme.Primary}
						size={Size.REGULAR}
						type="button"
						onClick={() => {
							void handleExport();
						}}
						disabled={selectedDbIds.size === 0 || exporting || databasesLoading}
						loading={exporting}
						iconPosition="left"
					>
						<Icon name={IconName.Download} className="h-4 w-4" />
						{exporting ? 'Exporting…' : 'Export'}
					</Button>
				</div>
			</section>

			<section className="rounded-lg border border-zinc-200 p-5 dark:border-zinc-700">
				<h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
					Import model
				</h2>
				<p className="mt-1 text-xs text-zinc-500">
					Upload a GSF model YAML file to write it into the catalog and semantic layer.
				</p>

				<FileUpload
					value={importFile}
					onChange={handleImportFileChange}
					onError={setImportError}
					validate={validateYamlFile}
					accept=".yaml,.yml"
					description="YAML files (.yaml, .yml)"
					aria-label="Upload a GSF model YAML file"
					className="mt-4"
				/>

				{importError && !confirmImportOpen ? (
					<p className="mt-2 text-sm text-red-600 dark:text-red-400">{importError}</p>
				) : null}

				<div className="mt-4 flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
					<div className="flex flex-col">
						<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
							Replace existing data
						</span>
						<span className="text-xs text-zinc-500">
							Deletes semantic-layer data for the imported databases that isn&apos;t
							in this file.
						</span>
					</div>
					<Toggle
						checked={replace}
						aria-label="Replace existing data"
						onChange={setReplace}
					/>
				</div>

				<div className="mt-3 flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 dark:border-zinc-700">
					<div className="flex flex-col">
						<span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
							Regenerate embeddings
						</span>
						<span className="text-xs text-zinc-500">
							Embeds newly created nodes and pushes them into the vector databases.
						</span>
					</div>
					<Toggle
						checked={embed}
						aria-label="Regenerate embeddings"
						onChange={setEmbed}
					/>
				</div>

				{importSummary ? (
					<div className="mt-3 rounded-lg border border-[#76b900]/30 bg-[#76b900]/5 p-3 text-xs text-zinc-700 dark:text-zinc-300">
						<p className="font-medium text-zinc-900 dark:text-zinc-100">
							Import summary
						</p>
						<ul className="mt-1.5 space-y-0.5">
							{createdCountEntries(importSummary.created).map(([key, value]) => (
								<li key={key}>
									Created {value} {key.replace(/_/g, ' ')}
								</li>
							))}
							{importSummary.embeddings ? (
								<li>
									{importSummary.embeddings.skipped
										? 'Embeddings skipped (embed API key not configured).'
										: `Embedded ${importSummary.embeddings.data_rows} data row(s) and ${importSummary.embeddings.semantic_rows} semantic row(s).`}
								</li>
							) : null}
						</ul>
					</div>
				) : null}

				<div className="mt-4 flex justify-end">
					<Button
						theme={ButtonTheme.Primary}
						size={Size.REGULAR}
						type="button"
						onClick={handleRequestImport}
						disabled={!importFile || importing}
						iconPosition="left"
					>
						<Icon name={IconName.Upload} className="h-4 w-4" />
						Import
					</Button>
				</div>
			</section>

			<ConfirmModal
				open={confirmImportOpen}
				title="Import model"
				message={
					<>
						This writes <strong>{importFile?.name}</strong> into the catalog and
						semantic layer.
						{replace ? (
							<>
								Because <strong>Replace existing data </strong> is on,
								semantic-layer data for the imported databases that isn&apos;t in
								this file will be deleted. This action cannot be undone.
							</>
						) : null}
					</>
				}
				confirmLabel="Import"
				tone="danger"
				onConfirm={handleConfirmImport}
				onCancel={handleCancelImport}
				confirming={importing}
				error={importError}
			/>

			<Toast
				open={exportMessage !== null}
				message={exportMessage ?? ''}
				variant="success"
				onClose={() => setExportMessage(null)}
			/>
		</div>
	);
};
