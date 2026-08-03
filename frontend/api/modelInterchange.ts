// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ResponseWithError } from './types';
import type { ImportModelResult } from '@/types/modelInterchange';

export type ImportModelOptions = {
	replace: boolean;
	embed: boolean;
};

export const modelInterchangeApi = {
	/** Export the scoped GSF model (Neo4j catalog + semantic layer) as a YAML file blob. */
	exportModel: (databaseIds: string[]): Promise<ResponseWithError<{ blob: Blob }>> =>
		requests.postBlob('model/export', { databases: databaseIds }),

	/** Import a GSF model YAML file, applying it to Neo4j and optionally refreshing VDB embeddings. */
	importModel: (
		file: File,
		{ replace, embed }: ImportModelOptions,
	): Promise<ResponseWithError<ImportModelResult>> => {
		const formData = new FormData();
		formData.append('file', file);
		return requests.postForm<ImportModelResult>('model/import', formData, { replace, embed });
	},
};
