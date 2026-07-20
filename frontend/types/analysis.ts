// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type CustomAnalysis = {
	id: string;
	name: string;
	description: string;
	sql: string;
};

export type PqlAnalysis = {
	id: string;
	name: string;
	description: string;
	pql: string;
};
