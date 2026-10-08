// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Lifecycle of an unused invite link. Accepted invites are deleted. */
export enum InvitationStatus {
	Active = 'active',
	Expired = 'expired',
}
