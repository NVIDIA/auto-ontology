import { SkeletonVariant } from '@/enums/skeleton';

type SkeletonBlockProps = {
	className?: string;
	variant?: SkeletonVariant;
};

export const SkeletonBlock = ({
	className = '',
	variant = SkeletonVariant.TEXT,
}: SkeletonBlockProps) => (
	<div
		aria-hidden
		className={`animate-pulse bg-zinc-100 dark:bg-zinc-800/70 ${
			variant === SkeletonVariant.CIRCLE
				? 'rounded-full'
				: variant === SkeletonVariant.RECTANGLE
					? 'rounded-lg'
					: 'rounded-md'
		} ${className}`}
	/>
);

type SkeletonCardProps = {
	className?: string;
	rows?: number;
};

export const SkeletonCard = ({ className = '', rows = 3 }: SkeletonCardProps) => (
	<div
		aria-hidden
		className={`rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900/50 ${className}`}
	>
		<div className="flex items-center gap-3">
			<SkeletonBlock variant={SkeletonVariant.CIRCLE} className="h-10 w-10 shrink-0" />
			<div className="flex-1 space-y-2">
				<SkeletonBlock className="h-4 w-2/5" />
				<SkeletonBlock className="h-3 w-3/5" />
			</div>
		</div>
		<div className="mt-5 space-y-3">
			{Array.from({ length: rows }).map((_, index) => (
				<SkeletonBlock
					key={index}
					className={`h-3 ${index === rows - 1 ? 'w-2/3' : 'w-full'}`}
				/>
			))}
		</div>
	</div>
);

type SkeletonRowsProps = {
	rows: number;
};

export const SkeletonRows = ({ rows }: SkeletonRowsProps) => (
	<div className="space-y-3" aria-hidden>
		{Array.from({ length: rows }).map((_, index) => (
			<div key={index} className="space-y-2 px-3 py-2">
				<SkeletonBlock className="h-3 w-3/4" />
				<SkeletonBlock className="h-2.5 w-1/3" />
			</div>
		))}
	</div>
);

type SkeletonTableProps = {
	columns?: number;
	rows?: number;
};

export const SkeletonTable = ({ columns = 4, rows = 8 }: SkeletonTableProps) => (
	<div
		aria-hidden
		className="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700"
	>
		<div className="grid grid-flow-col auto-cols-fr gap-4 bg-zinc-100 px-4 py-3 dark:bg-zinc-800">
			{Array.from({ length: columns }).map((_, index) => (
				<SkeletonBlock key={index} className="h-3" />
			))}
		</div>
		{Array.from({ length: rows }).map((_, rowIndex) => (
			<div
				key={rowIndex}
				className="grid grid-flow-col auto-cols-fr gap-4 border-t border-zinc-100 px-4 py-3 dark:border-zinc-800"
			>
				{Array.from({ length: columns }).map((_, columnIndex) => (
					<SkeletonBlock key={columnIndex} className="h-3" />
				))}
			</div>
		))}
	</div>
);

type SkeletonSqlBlocksProps = {
	items?: number;
	/** Adds a name and description line above each block, as `SqlAttributesModal` shows. */
	withHeading?: boolean;
};

/** Placeholder for a list of `SqlBlock`s: a label bar over a few lines of code. */
export const SkeletonSqlBlocks = ({ items = 3, withHeading = false }: SkeletonSqlBlocksProps) => (
	<div className="space-y-4" aria-hidden>
		{Array.from({ length: items }).map((_, index) => (
			<div
				key={index}
				className={
					withHeading
						? 'rounded-lg border border-zinc-200 p-3 dark:border-zinc-700'
						: undefined
				}
			>
				{withHeading ? (
					<div className="mb-2 space-y-2">
						<SkeletonBlock className="h-4 w-40" />
						<SkeletonBlock className="h-3 w-64" />
					</div>
				) : null}
				<div className="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-700">
					<div className="flex items-center justify-between border-b border-zinc-200 px-3 py-2 dark:border-zinc-700">
						<SkeletonBlock className="h-3 w-16" />
						<SkeletonBlock className="h-3 w-3" />
					</div>
					<div className="space-y-2 p-3">
						<SkeletonBlock className="h-3 w-full" />
						<SkeletonBlock className="h-3 w-5/6" />
						<SkeletonBlock className="h-3 w-2/3" />
					</div>
				</div>
			</div>
		))}
	</div>
);

export const SkeletonDetail = () => (
	<div className="flex flex-1 flex-col gap-6 p-6" aria-hidden>
		<div className="space-y-3">
			<SkeletonBlock className="h-7 w-64" />
			<SkeletonBlock className="h-4 w-2/5" />
		</div>
		<div className="grid grid-cols-2 gap-4">
			{Array.from({ length: 4 }).map((_, index) => (
				<div key={index} className="space-y-2">
					<SkeletonBlock className="h-3 w-1/3" />
					<SkeletonBlock className="h-5 w-full" />
				</div>
			))}
		</div>
		<SkeletonBlock variant={SkeletonVariant.RECTANGLE} className="h-36 w-full" />
		<SkeletonTable columns={3} rows={4} />
	</div>
);
