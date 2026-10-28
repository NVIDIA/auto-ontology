// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';

import { Button } from '@/common/Button';
import { Icon, IconName } from '@/common/icons';
import { Label } from '@/common/Label';
import { OwnerLabel } from '@/common/OwnerLabel';
import { OwnersSelect } from '@/common/OwnersSelect';
import { PopoverMenu } from '@/common/PopoverMenu';
import { formatDate } from '@/common/date';
import { AUTO_GENERATED_LABEL } from '@/constants/tags';
import { Size, ButtonTheme } from '@/enums/button';
import { ownerChoices } from '@/lib/owners';
import { ownerQueries } from '@/lib/queries/owners';
import type {
	ComposerEntityTagsSection,
	ComposerOwnerOption,
	ComposerOwnerSection,
	ComposerTagChip,
} from '@/types/composer-section';
import type { Owner } from '@/types/owners';
import type { TagAuthor } from '@/types/tags';
import type { GlossaryActivity } from '@/types/terms';

export const SIDE_PANEL_WIDTH = '280px';

export type SidePanelActivity = {
	kind: 'activity';
	title: string;
	author: TagAuthor | null;
	date: string | null;
};

export type SidePanelBulk =
	| { kind: 'owner'; section: ComposerOwnerSection }
	| { kind: 'tags'; section: ComposerEntityTagsSection }
	| SidePanelActivity;

const byName = (a: ComposerTagChip, b: ComposerTagChip) =>
	a.name.toLowerCase().localeCompare(b.name.toLowerCase()) || a.id.localeCompare(b.id);

const ownerOption = (owner: Owner): ComposerOwnerOption => ({
	id: owner.id,
	name: owner.name,
	kind: owner.kind,
	color: owner.color,
});

/**
 * Owner and tags for a detail page, on the left or the right.
 * Pass `activity` for Last Modified. Created By is only for a row a person
 * can insert, such as a SQL attribute.
 */
export const entitySidePanel = ({
	owner,
	tags,
	activity,
	showCreatedBy = false,
}: {
	owner: ComposerOwnerSection;
	tags: ComposerEntityTagsSection;
	activity?: GlossaryActivity;
	showCreatedBy?: boolean;
}): { bulks: SidePanelBulk[]; width: string } => ({
	width: SIDE_PANEL_WIDTH,
	bulks: [
		{ kind: 'owner', section: owner },
		...(showCreatedBy
			? [
					{
						kind: 'activity' as const,
						title: 'Created By',
						author: activity?.created_by_user ?? null,
						date: activity?.created ?? null,
					},
				]
			: []),
		...(activity != null
			? [
					{
						kind: 'activity' as const,
						title: 'Last Modified',
						author: activity.modified_by_user ?? null,
						date: activity.modified ?? null,
					},
				]
			: []),
		{ kind: 'tags', section: tags },
	],
});

const useLiveOwnerOptions = (section: ComposerOwnerSection): ComposerOwnerOption[] => {
	const { data } = useQuery(ownerQueries.all());
	if (data == null) return section.options;
	const source =
		section.editable === false ? data : ownerChoices(data, section.permission, section.ownerId);
	return source.map(ownerOption);
};

const PanelBlock = ({
	title,
	editing = false,
	children,
}: {
	title: string;
	editing?: boolean;
	children: ReactNode;
}) => (
	<section className="min-w-0">
		<h2 className="mb-2 text-sm font-semibold text-heading dark:text-zinc-100">{title}</h2>
		<div
			className={`min-w-0 rounded-lg border bg-white p-3.5 dark:bg-zinc-950 ${
				editing
					? 'border-[#76b900]/60 ring-1 ring-[#76b900]/10'
					: 'border-zinc-200/90 dark:border-zinc-700/90'
			}`}
		>
			{children}
		</div>
	</section>
);

const CalendarIcon = () => (
	<svg
		viewBox="0 0 24 24"
		fill="none"
		stroke="currentColor"
		strokeWidth="1.5"
		aria-hidden="true"
		className="h-4 w-4 shrink-0 text-secondary dark:text-zinc-400"
	>
		<path
			strokeLinecap="round"
			strokeLinejoin="round"
			d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5"
		/>
	</svg>
);

const ActivityBody = ({ bulk }: { bulk: SidePanelActivity }) => {
	const name = bulk.author?.name || bulk.author?.email || '';
	const neverModified = bulk.title === 'Last Modified' && name === '' && bulk.date == null;
	if (neverModified) {
		return <p className="text-sm text-secondary dark:text-zinc-400">—</p>;
	}
	const label = name || AUTO_GENERATED_LABEL;
	return (
		<div className="space-y-3">
			<span className="flex min-w-0 items-center gap-2 text-sm text-body dark:text-zinc-300">
				<span
					aria-hidden="true"
					className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${
						name === ''
							? 'bg-zinc-200 text-secondary dark:bg-zinc-700 dark:text-zinc-400'
							: 'bg-[#76b900] text-white'
					}`}
				>
					{label.charAt(0).toUpperCase()}
				</span>
				<span className="truncate">{label}</span>
			</span>
			{bulk.date != null && (
				<span className="flex items-center gap-2 text-sm text-body dark:text-zinc-300">
					<CalendarIcon />
					{formatDate(bulk.date, 'MMM DD, YYYY')}
				</span>
			)}
		</div>
	);
};

const OwnerEditor = ({
	section,
	options,
	onChange,
}: {
	section: ComposerOwnerSection;
	options: ComposerOwnerOption[];
	onChange: (id: string, value: string) => void;
}) => {
	const [ownerId, setOwnerId] = useState(section.ownerId ?? '');
	return (
		<OwnersSelect
			owners={options}
			value={ownerId}
			onChange={(next) => {
				setOwnerId(next);
				onChange(section.id, next);
			}}
		/>
	);
};

const OwnerBulk = ({
	section,
	editing,
	onChange,
}: {
	section: ComposerOwnerSection;
	editing: boolean;
	onChange: (id: string, value: string) => void;
}) => {
	const options = useLiveOwnerOptions(section);
	const canEdit = editing && section.editable !== false;
	const owner = options.find((option) => option.id === section.ownerId);
	return (
		<PanelBlock title={section.title} editing={canEdit}>
			{canEdit ? (
				<OwnerEditor section={section} options={options} onChange={onChange} />
			) : (
				<OwnerLabel owner={owner} />
			)}
		</PanelBlock>
	);
};

const TagEditor = ({
	section,
	onChange,
}: {
	section: ComposerEntityTagsSection;
	onChange: (id: string, value: string[]) => void;
}) => {
	const [tags, setTags] = useState<ComposerTagChip[]>(() => [...section.tags].sort(byName));
	const assigned = new Set(tags.map((tag) => tag.id));
	const unassigned = section.options.filter((tag) => !assigned.has(tag.id));

	const apply = (next: ComposerTagChip[]) => {
		setTags(next);
		onChange(
			section.id,
			next.map((tag) => tag.id),
		);
	};

	return (
		<>
			<div className="mb-2 flex justify-end">
				<PopoverMenu
					items={unassigned.map((tag) => ({
						label: tag.name,
						onClick: () => apply([...tags, tag].sort(byName)),
					}))}
					header={
						unassigned.length === 0 ? (
							<p className="px-3 py-1.5 text-xs text-secondary dark:text-zinc-400">
								{section.options.length === 0
									? 'No tags exist yet'
									: 'Every tag is already applied'}
							</p>
						) : undefined
					}
					trigger={({ toggle }) => (
						<Button
							theme={ButtonTheme.Soft}
							size={Size.SMALL}
							type="button"
							onClick={toggle}
							iconPosition="left"
						>
							<Icon name={IconName.Plus} className="h-3.5 w-3.5" />
							Add Tag
						</Button>
					)}
				/>
			</div>
			{tags.length === 0 ? (
				<p className="text-sm text-secondary dark:text-zinc-400">No Tags Added</p>
			) : (
				<ul className="flex w-full min-w-0 flex-wrap gap-1.5">
					{tags.map((tag) => (
						<li key={tag.id} className="flex max-w-full min-w-0">
							<Label
								label={tag.name}
								onRemove={() => apply(tags.filter((item) => item.id !== tag.id))}
							/>
						</li>
					))}
				</ul>
			)}
		</>
	);
};

const TagList = ({ tags }: { tags: ComposerTagChip[] }) =>
	tags.length === 0 ? (
		<p className="text-sm text-secondary dark:text-zinc-400">No Tags Added</p>
	) : (
		<ul className="flex w-full min-w-0 flex-wrap gap-1.5">
			{[...tags].sort(byName).map((tag) => (
				<li key={tag.id} className="flex max-w-full min-w-0">
					<Label label={tag.name} />
				</li>
			))}
		</ul>
	);

const TagBulk = ({
	section,
	editing,
	onChange,
}: {
	section: ComposerEntityTagsSection;
	editing: boolean;
	onChange: (id: string, value: string[]) => void;
}) => {
	const canEdit = editing && section.editable !== false;
	return (
		<PanelBlock title={section.title} editing={canEdit}>
			{canEdit ? (
				<TagEditor section={section} onChange={onChange} />
			) : (
				<TagList tags={section.tags} />
			)}
		</PanelBlock>
	);
};

export const SidePanel = ({
	bulks,
	isEditing,
	onChange,
}: {
	bulks: SidePanelBulk[];
	isEditing: boolean;
	onChange: (id: string, value: string | string[]) => void;
}) => (
	<div className="flex w-full min-w-0 flex-col gap-5 px-4 py-5">
		{bulks.map((bulk) => {
			if (bulk.kind === 'owner') {
				return (
					<OwnerBulk
						key={bulk.section.id}
						section={bulk.section}
						editing={isEditing}
						onChange={onChange}
					/>
				);
			}
			if (bulk.kind === 'tags') {
				return (
					<TagBulk
						key={bulk.section.id}
						section={bulk.section}
						editing={isEditing}
						onChange={onChange}
					/>
				);
			}
			return (
				<PanelBlock key={bulk.title} title={bulk.title}>
					<ActivityBody bulk={bulk} />
				</PanelBlock>
			);
		})}
	</div>
);
