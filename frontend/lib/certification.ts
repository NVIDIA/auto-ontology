// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CertificationStatus } from '@/enums/certification';

/** Shape shared by attributes, which carry a single top-level certification flag. */
export type AttributeCertifiableFields = {
	certified: boolean;
};

/** Binary status for a single boolean flag. */
export const fieldStatus = (certified: boolean): CertificationStatus =>
	certified ? CertificationStatus.Certified : CertificationStatus.Pending;

/**
 * Binary status for one attribute (column or sql): CERTIFIED when its single
 * `certified` flag is set, otherwise PENDING. Never PARTIAL.
 */
export const attributeStatus = (item: AttributeCertifiableFields): CertificationStatus =>
	fieldStatus(item.certified);

/*
 * A term's three-state aggregate status is deliberately NOT computed here.
 * The Terms list only receives per-term attribute *counts*, never the
 * attributes themselves, so the rollup can only be done server-side — see
 * `_certification` in gsf/dal/terms.py. Every read returns it as
 * `Term.certification` and every certification write returns the recomputed
 * value, so there is exactly one implementation of the rule.
 */
