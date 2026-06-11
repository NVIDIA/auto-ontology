// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FC, SVGProps } from 'react';

import MenuSvg from './svg/menu.svg';
import DotsVerticalSvg from './svg/dots-vertical.svg';
import PencilSvg from './svg/pencil.svg';
import TrashSvg from './svg/trash.svg';
import SendSvg from './svg/send.svg';
import StopSvg from './svg/stop.svg';
import CheckSvg from './svg/check.svg';
import ChatBubbleSvg from './svg/chat-bubble.svg';
import ChevronRightSvg from './svg/chevron-right.svg';
import NvidiaLogoSvg from './svg/nvidia-logo.svg';
import DatabaseSvg from './svg/database.svg';
import SettingsSvg from './svg/settings.svg';
import ChartBarSvg from './svg/chart-bar.svg';
import TableSvg from './svg/table.svg';
import ViewSvg from './svg/view.svg';
import MaterializedViewSvg from './svg/materialized-view.svg';
import SchemaSvg from './svg/schema.svg';
import ColumnSvg from './svg/column.svg';

export enum IconName {
	Menu = 'menu',
	DotsVertical = 'dots-vertical',
	Pencil = 'pencil',
	Trash = 'trash',
	Send = 'send',
	Stop = 'stop',
	Check = 'check',
	ChatBubble = 'chat-bubble',
	ChevronRight = 'chevron-right',
	NvidiaLogo = 'nvidia-logo',
	Database = 'database',
	Settings = 'settings',
	ChartBar = 'chart-bar',
	Table = 'table',
	View = 'view',
	MaterializedView = 'materialized-view',
	Schema = 'schema',
	Column = 'column',
}

const icons: Record<IconName, FC<SVGProps<SVGSVGElement>>> = {
	[IconName.Menu]: MenuSvg,
	[IconName.DotsVertical]: DotsVerticalSvg,
	[IconName.Pencil]: PencilSvg,
	[IconName.Trash]: TrashSvg,
	[IconName.Send]: SendSvg,
	[IconName.Stop]: StopSvg,
	[IconName.Check]: CheckSvg,
	[IconName.ChatBubble]: ChatBubbleSvg,
	[IconName.ChevronRight]: ChevronRightSvg,
	[IconName.NvidiaLogo]: NvidiaLogoSvg,
	[IconName.Database]: DatabaseSvg,
	[IconName.Settings]: SettingsSvg,
	[IconName.ChartBar]: ChartBarSvg,
	[IconName.Table]: TableSvg,
	[IconName.View]: ViewSvg,
	[IconName.MaterializedView]: MaterializedViewSvg,
	[IconName.Schema]: SchemaSvg,
	[IconName.Column]: ColumnSvg,
};

export type IconProps = SVGProps<SVGSVGElement> & {
	name: IconName;
};

export const Icon: FC<IconProps> = ({ name, ...props }) => {
	const SvgComponent = icons[name];
	return <SvgComponent {...props} />;
};
