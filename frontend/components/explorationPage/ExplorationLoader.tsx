'use client';

import Image from 'next/image';

type ExplorationLoaderProps = {
	overlay?: boolean;
};

export const ExplorationLoader = ({ overlay = false }: ExplorationLoaderProps) => (
	<div
		className={`flex h-full w-full items-center justify-center bg-zinc-50 dark:bg-zinc-950${
			overlay ? ' absolute inset-0 z-10' : ''
		}`}
		role="status"
		aria-label="Loading exploration"
	>
		<Image
			src="/exploration-loader.svg"
			alt=""
			width={720}
			height={430}
			unoptimized
			loading="eager"
			className="h-auto w-full max-w-[720px]"
		/>
	</div>
);
