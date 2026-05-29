// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared catalog building blocks.
 *
 * These encode the information architecture we borrowed from illumex's semantic
 * layer (titled section cards, a certification status, a usage meter, coverage
 * bars and counted category tabs) while staying in the GSF / NVIDIA visual
 * language: zinc neutrals, the #76b900 accent, rounded cards and dark mode.
 */

import type { ReactNode } from 'react';
import type { OmTable, OmUsageSummary } from '@/types/openmetadata';

/* ------------------------------------------------------------------ icons -- */

type IconProps = { className?: string };

export const DocIcon = ({ className }: IconProps) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className={className} aria-hidden>
		<path d="M4 4a2 2 0 0 1 2-2h5.586A2 2 0 0 1 13 2.586L15.414 5A2 2 0 0 1 16 6.414V16a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4zm3 5a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2H7zm0 4a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2H7z" />
	</svg>
);

export const InfoIcon = ({ className }: IconProps) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className={className} aria-hidden>
		<path
			fillRule="evenodd"
			d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0zm-7-4a1 1 0 1 1-2 0 1 1 0 0 1 2 0zM9 9a1 1 0 0 0 0 2v3a1 1 0 0 0 1 1h1a1 1 0 1 0 0-2v-3a1 1 0 0 0-1-1H9z"
			clipRule="evenodd"
		/>
	</svg>
);

export const StatsIcon = ({ className }: IconProps) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className={className} aria-hidden>
		<path d="M3 12a1 1 0 0 1 1 1v3a1 1 0 1 1-2 0v-3a1 1 0 0 1 1-1zm5-5a1 1 0 0 1 1 1v8a1 1 0 1 1-2 0V8a1 1 0 0 1 1-1zm5-3a1 1 0 0 1 1 1v11a1 1 0 1 1-2 0V5a1 1 0 0 1 1-1zm5 6a1 1 0 0 1 1 1v5a1 1 0 1 1-2 0v-5a1 1 0 0 1 1-1z" />
	</svg>
);

export const FireIcon = ({ className }: IconProps) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className={className} aria-hidden>
		<path
			fillRule="evenodd"
			d="M12.395 2.553a1 1 0 0 0-1.45-.385c-.345.23-.614.558-.822.88-.214.33-.403.713-.57 1.116-.334.804-.614 1.768-.84 2.734a31.365 31.365 0 0 0-.613 3.58 2.64 2.64 0 0 1-.945-1.067c-.328-.68-.398-1.534-.398-2.654A1 1 0 0 0 5.05 6.05 6.981 6.981 0 0 0 3 11a7 7 0 1 0 11.95-4.95c-.592-.591-.98-.985-1.348-1.467-.363-.476-.724-1.063-1.207-2.03zM12.12 15.12A3 3 0 0 1 7 13s.879.5 2.5.5c0-1 .5-4 1.25-4.5.5 1 .786 1.293 1.371 1.879A2.99 2.99 0 0 1 13 13a2.99 2.99 0 0 1-.879 2.121z"
			clipRule="evenodd"
		/>
	</svg>
);

export const LinkIcon = ({ className }: IconProps) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className={className} aria-hidden>
		<path d="M11 3a1 1 0 1 0 0 2h2.586l-6.293 6.293a1 1 0 1 0 1.414 1.414L15 6.414V9a1 1 0 1 0 2 0V4a1 1 0 0 0-1-1h-5z" />
		<path d="M5 5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-3a1 1 0 1 0-2 0v3H5V7h3a1 1 0 0 0 0-2H5z" />
	</svg>
);

export const UserIcon = ({ className }: IconProps) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className={className} aria-hidden>
		<path
			fillRule="evenodd"
			d="M10 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm-7 9a7 7 0 1 1 14 0H3z"
			clipRule="evenodd"
		/>
	</svg>
);

export const GridIcon = ({ className }: IconProps) => (
	<svg viewBox="0 0 20 20" fill="currentColor" className={className} aria-hidden>
		<path d="M5 3a2 2 0 0 0-2 2v2a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H5zm8 0a2 2 0 0 0-2 2v2a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2h-2zM5 11a2 2 0 0 0-2 2v2a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-2a2 2 0 0 0-2-2H5zm8 0a2 2 0 0 0-2 2v2a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2v-2a2 2 0 0 0-2-2h-2z" />
	</svg>
);

/* ------------------------------------------------------------------ panel -- */

export const Panel = ({
	title,
	icon,
	action,
	children,
	className = '',
}: {
	title?: string;
	icon?: ReactNode;
	action?: ReactNode;
	children: ReactNode;
	className?: string;
}) => (
	<section
		className={`rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900 ${className}`}
	>
		{title ? (
			<header className="flex items-center justify-between gap-3 border-b border-zinc-100 px-4 py-2.5 dark:border-zinc-800">
				<h3 className="flex items-center gap-2 text-[13px] font-semibold text-zinc-700 dark:text-zinc-200">
					{icon ? <span className="text-zinc-400 dark:text-zinc-500">{icon}</span> : null}
					{title}
				</h3>
				{action}
			</header>
		) : null}
		{children}
	</section>
);

export const MetaField = ({
	label,
	children,
	mono = false,
}: {
	label: string;
	children: ReactNode;
	mono?: boolean;
}) => (
	<div className="flex items-start justify-between gap-3 px-4 py-2.5">
		<span className="shrink-0 text-xs text-zinc-500 dark:text-zinc-500">{label}</span>
		<span
			className={`min-w-0 text-right text-sm text-zinc-800 dark:text-zinc-200 ${
				mono ? 'font-mono text-xs' : ''
			}`}
		>
			{children}
		</span>
	</div>
);

/* ---------------------------------------------------------- certification -- */

export type Readiness = 'certified' | 'partial' | 'pending';

/**
 * Derive an illumex-style certification status from the metadata we actually
 * harvest: a table is "Certified" when it is documented, owned and has tags,
 * "Partial" when at least one of those is present, otherwise "Pending".
 */
export const readinessOf = (t: OmTable): Readiness => {
	const documented = Boolean(t.description && t.description.trim().length > 0);
	const owned = Boolean(t.owners && t.owners.length > 0);
	const tagged = Boolean(t.tags && t.tags.length > 0);
	const score = Number(documented) + Number(owned) + Number(tagged);
	if (score >= 3) return 'certified';
	if (score >= 1) return 'partial';
	return 'pending';
};

const READINESS_META: Record<Readiness, { label: string; cls: string; dot: string }> = {
	certified: {
		label: 'Certified',
		cls: 'bg-emerald-100 text-emerald-800 ring-emerald-300 dark:bg-emerald-950/50 dark:text-emerald-200 dark:ring-emerald-800',
		dot: 'bg-emerald-500',
	},
	partial: {
		label: 'Partial',
		cls: 'bg-amber-100 text-amber-900 ring-amber-300 dark:bg-amber-950/50 dark:text-amber-200 dark:ring-amber-800',
		dot: 'bg-amber-500',
	},
	pending: {
		label: 'Pending',
		cls: 'bg-zinc-100 text-zinc-600 ring-zinc-300 dark:bg-zinc-800 dark:text-zinc-400 dark:ring-zinc-700',
		dot: 'bg-zinc-400',
	},
};

export const CertificationBadge = ({
	status,
	dense = false,
}: {
	status: Readiness;
	dense?: boolean;
}) => {
	const m = READINESS_META[status];
	return (
		<span
			className={`inline-flex items-center gap-1.5 rounded-full ring-1 ring-inset ${m.cls} ${
				dense ? 'px-1.5 py-0 text-[10px]' : 'px-2 py-0.5 text-xs'
			} font-medium`}
		>
			<span className={`h-1.5 w-1.5 rounded-full ${m.dot}`} aria-hidden />
			{m.label}
		</span>
	);
};

/* ------------------------------------------------------------ usage meter -- */

export type UsageLevel = 'high' | 'medium' | 'low' | 'none';

export const usageLevelOf = (summary: OmUsageSummary | undefined): UsageLevel => {
	const rank =
		summary?.weeklyStats?.percentileRank ??
		summary?.monthlyStats?.percentileRank ??
		summary?.dailyStats?.percentileRank;
	if (rank == null) return 'none';
	if (rank >= 66) return 'high';
	if (rank >= 33) return 'medium';
	return 'low';
};

const USAGE_META: Record<UsageLevel, { label: string; bars: number; cls: string }> = {
	high: { label: 'High', bars: 3, cls: 'text-emerald-600 dark:text-emerald-400' },
	medium: { label: 'Medium', bars: 2, cls: 'text-amber-600 dark:text-amber-400' },
	low: { label: 'Low', bars: 1, cls: 'text-zinc-500 dark:text-zinc-400' },
	none: { label: 'No data', bars: 0, cls: 'text-zinc-400 dark:text-zinc-600' },
};

export const UsageMeter = ({ level }: { level: UsageLevel }) => {
	const m = USAGE_META[level];
	return (
		<span className={`inline-flex items-center gap-1.5 text-xs font-medium ${m.cls}`}>
			<span className="flex items-end gap-0.5" aria-hidden>
				{[0, 1, 2].map((i) => (
					<span
						key={i}
						className={`w-1 rounded-sm ${i < m.bars ? 'bg-current' : 'bg-zinc-200 dark:bg-zinc-700'}`}
						style={{ height: `${4 + i * 3}px` }}
					/>
				))}
			</span>
			{m.label}
		</span>
	);
};

/* ---------------------------------------------------------- coverage bars -- */

export const CoverageBar = ({
	label,
	value,
	total,
	tone = 'green',
}: {
	label: string;
	value: number;
	total: number;
	tone?: 'green' | 'amber' | 'sky' | 'violet';
}) => {
	const pct = total > 0 ? Math.round((value / total) * 100) : 0;
	const fill =
		tone === 'amber'
			? 'bg-amber-500'
			: tone === 'sky'
				? 'bg-sky-500'
				: tone === 'violet'
					? 'bg-violet-500'
					: 'bg-[#76b900]';
	return (
		<div>
			<div className="mb-1.5 flex items-baseline justify-between">
				<span className="text-xs font-medium text-zinc-600 dark:text-zinc-300">
					{label}
				</span>
				<span className="text-xs tabular-nums text-zinc-500 dark:text-zinc-400">
					{value.toLocaleString()} / {total.toLocaleString()}
					<span className="ml-1.5 font-semibold text-zinc-700 dark:text-zinc-200">
						{pct}%
					</span>
				</span>
			</div>
			<div className="h-2 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
				<div className={`h-full rounded-full ${fill}`} style={{ width: `${pct}%` }} />
			</div>
		</div>
	);
};

/* ---------------------------------------------------------- category tabs -- */

export type CategoryTab<T extends string> = { id: T; label: string; count?: number };

export const CategoryTabs = <T extends string>({
	tabs,
	active,
	onSelect,
}: {
	tabs: ReadonlyArray<CategoryTab<T>>;
	active: T;
	onSelect: (id: T) => void;
}) => (
	<div className="flex flex-wrap items-center gap-1 border-b border-zinc-200 dark:border-zinc-800">
		{tabs.map((t) => {
			const isActive = t.id === active;
			return (
				<button
					key={t.id}
					type="button"
					onClick={() => onSelect(t.id)}
					className={`relative flex items-center gap-1.5 px-3 py-2 text-sm font-medium transition-colors ${
						isActive
							? 'text-[#76b900]'
							: 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-200'
					}`}
				>
					{t.label}
					{t.count != null ? (
						<span
							className={`rounded-full px-1.5 py-px text-[10px] tabular-nums ${
								isActive
									? 'bg-[#76b900]/15 text-[#76b900]'
									: 'bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400'
							}`}
						>
							{t.count}
						</span>
					) : null}
					{isActive ? (
						<span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-[#76b900]" />
					) : null}
				</button>
			);
		})}
	</div>
);

/* ---------------------------------------------------------------- avatar -- */

const AVATAR_TONES = [
	'bg-sky-100 text-sky-700 dark:bg-sky-950/50 dark:text-sky-300',
	'bg-violet-100 text-violet-700 dark:bg-violet-950/50 dark:text-violet-300',
	'bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-300',
	'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300',
	'bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300',
];

const initialsOf = (name: string): string =>
	name
		.split(/[\s._-]+/)
		.filter(Boolean)
		.slice(0, 2)
		.map((p) => p[0]?.toUpperCase() ?? '')
		.join('') || '?';

export const OwnerChip = ({ name }: { name: string | null | undefined }) => {
	if (!name) {
		return <span className="text-xs text-zinc-400 dark:text-zinc-600">unassigned</span>;
	}
	let hash = 0;
	for (let i = 0; i < name.length; i += 1)
		hash = (hash + name.charCodeAt(i)) % AVATAR_TONES.length;
	return (
		<span className="inline-flex items-center gap-1.5">
			<span
				className={`flex h-5 w-5 items-center justify-center rounded-full text-[9px] font-semibold ${AVATAR_TONES[hash]}`}
				aria-hidden
			>
				{initialsOf(name)}
			</span>
			<span className="truncate text-sm text-zinc-700 dark:text-zinc-300">{name}</span>
		</span>
	);
};
