// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Icon, IconName } from '@/common/icons';
import { getOpenInvitation } from '@/lib/invitations';
import { InviteSetPassword } from './InviteSetPassword';

const InviteInvalid = () => (
	<div className="flex h-full w-full items-center justify-center bg-white p-6 dark:bg-zinc-950">
		<div className="flex max-w-sm flex-col items-center gap-4 text-center">
			<div className="flex items-center gap-2">
				<Icon name={IconName.NvidiaLogo} className="h-6 w-6" />
				<span className="text-lg font-semibold text-heading dark:text-zinc-100">
					Auto Ontology
				</span>
			</div>
			<p className="text-sm text-body dark:text-zinc-300">
				This invitation is invalid or has expired.
			</p>
		</div>
	</div>
);

const InvitePage = async ({ params }: { params: Promise<{ token: string }> }) => {
	const { token } = await params;
	const invitation = await getOpenInvitation(token);
	if (!invitation) return <InviteInvalid />;
	return <InviteSetPassword token={token} email={invitation.email} kind={invitation.kind} />;
};

export default InvitePage;
