// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pageQuery, requests } from './requests';
import type { ApiPagedResponse, PageParams, ResponseWithError } from './types';
import type { DocumentItem, DocumentUpdateInput } from '@/types/documents';

export type DocumentsListParams = PageParams & {
	/** Case-insensitive substring, matched against name and description. */
	query?: string;
};

export const documentsApi = {
	/** Documents, newest first, with `total` counting the whole match. */
	getAll: (params?: DocumentsListParams): Promise<ApiPagedResponse<DocumentItem[]>> =>
		requests.get('documents', {
			...(params?.query ? { query: params.query } : {}),
			...pageQuery(params),
		}),

	/**
	 * Upload one file. Answers as soon as it is stored, with the document
	 * `processing`; the list shows it turn `ready` or `failed`.
	 */
	upload: (file: File): Promise<ResponseWithError<{ data: DocumentItem }>> => {
		const formData = new FormData();
		formData.append('file', file);
		return requests.postForm<{ data: DocumentItem }>('documents', formData);
	},

	update: (
		documentId: string,
		input: DocumentUpdateInput,
	): Promise<ResponseWithError<{ data: DocumentItem }>> =>
		requests.patch(`documents/${documentId}`, input),

	/** Restart a failed document from its stored upload. */
	retry: (documentId: string): Promise<ResponseWithError<{ data: DocumentItem }>> =>
		requests.post(`documents/${documentId}/retry`),

	/** 404 when the document is already gone, which means the caller's list is stale. */
	delete: (documentId: string): Promise<ResponseWithError<{ data: { id: string } }>> =>
		requests.delete(`documents/${documentId}`),
};
