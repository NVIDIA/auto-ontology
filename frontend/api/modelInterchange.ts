// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ResponseWithError } from './types';
import type { ModelFormat } from '@/enums/modelInterchange';
import type { ImportModelResult } from '@/types/modelInterchange';

export type ImportModelOptions = {
	replace: boolean;
	embed: boolean;
};

export const modelInterchangeApi = {
	/**
	 * Export the scoped model (catalog + semantic layer) as a YAML file blob in `format`.
	 * With `includeOntology` (Apache Ossie only) the blob is a zip that also holds the
	 * terms as an Ossie ontology and its mapping onto the semantic model.
	 */
	exportModel: (
		databaseIds: string[],
		format: ModelFormat,
		includeOntology = false,
	): Promise<ResponseWithError<{ blob: Blob }>> =>
		requests.postBlob('model/export', {
			databases: databaseIds,
			format,
			include_ontology: includeOntology,
		}),

	/**
	 * Import a native Auto Ontology or Apache Ossie model YAML file, applying it and
	 * optionally refreshing VDB embeddings. The backend detects the format from the file.
	 */
	importModel: (
		file: File,
		{ replace, embed }: ImportModelOptions,
	): Promise<ResponseWithError<ImportModelResult>> => {
		const formData = new FormData();
		formData.append('file', file);
		return requests.postForm<ImportModelResult>('model/import', formData, { replace, embed });
	},
};
