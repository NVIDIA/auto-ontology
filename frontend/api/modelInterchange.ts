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
	/** Export the scoped model (Neo4j catalog + semantic layer) as a YAML file blob in `format`. */
	exportModel: (
		databaseIds: string[],
		format: ModelFormat,
	): Promise<ResponseWithError<{ blob: Blob }>> =>
		requests.postBlob('model/export', { databases: databaseIds, format }),

	/**
	 * Import a native GSF or Apache Ossie model YAML file, applying it to Neo4j and
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
