// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import axios, { AxiosError, AxiosResponse } from 'axios';
import { ApiError, ResponseWithError } from './types';

// Query params accepted by `requests.get`. Numbers/booleans are serialised by
// axios at request time, so callers can pass them without stringifying first.
export type QueryParams = Record<string, string | number | boolean | string[] | undefined>;

const isServer = typeof window === 'undefined';

const api = axios.create({
	baseURL: isServer ? `${process.env.PYTHON_API_URL ?? 'http://127.0.0.1:3001'}/api` : '/api',
	headers: {
		'Content-Type': 'application/json',
	},
});

const responseBody = <DataType>({ data }: AxiosResponse<DataType>): DataType => data;

const extractServerMessage = (data: unknown): string | undefined => {
	if (data == null || typeof data !== 'object') return undefined;
	const { detail } = data as { detail?: unknown };
	return typeof detail === 'string' ? detail : undefined;
};

const errorHandler = (err: AxiosError): ApiError => {
	return {
		message: (err.response != null && extractServerMessage(err.response.data)) || err.message,
		error: true,
	};
};

const normalizeQueryParams = (params: QueryParams): QueryParams => {
	const newParams: QueryParams = {};
	Object.entries(params).forEach(([paramKey, paramValue]) => {
		if (Array.isArray(paramValue)) {
			newParams[paramKey] = paramValue.toString();
		} else {
			newParams[paramKey] = paramValue;
		}
	});
	return newParams;
};

export const requests = {
	get: <OutputType>(url: string, params: QueryParams = {}, abortController?: AbortController) => {
		return api
			.get<OutputType>(url, {
				params: normalizeQueryParams(params),
				signal: abortController?.signal,
			})
			.then(responseBody)
			.catch(errorHandler) as Promise<ResponseWithError<OutputType>>;
	},

	post: <OutputType>(url: string, data: unknown = {}, abortController?: AbortController) => {
		return api
			.post<OutputType>(url, data, {
				signal: abortController?.signal,
			})
			.then(responseBody)
			.catch(errorHandler) as Promise<ResponseWithError<OutputType>>;
	},

	put: <OutputType>(url: string, data: unknown = {}, abortController?: AbortController) => {
		return api
			.put<OutputType>(url, data, {
				signal: abortController?.signal,
			})
			.then(responseBody)
			.catch(errorHandler) as Promise<ResponseWithError<OutputType>>;
	},

	patch: <OutputType>(url: string, body: Record<string, unknown> = {}) => {
		return api.patch<OutputType>(url, body).then(responseBody).catch(errorHandler) as Promise<
			ResponseWithError<OutputType>
		>;
	},

	delete: <OutputType>(url: string, abortController?: AbortController) => {
		return api
			.delete<OutputType>(url, {
				signal: abortController?.signal,
			})
			.then(responseBody)
			.catch(errorHandler) as Promise<ResponseWithError<OutputType>>;
	},
};
