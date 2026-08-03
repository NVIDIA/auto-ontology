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
import CopySvg from './svg/copy.svg';
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
import ChartLineSvg from './svg/chart-line.svg';
import UsersSvg from './svg/users.svg';
import KeySvg from './svg/key.svg';
import TermsSvg from './svg/terms.svg';
import ExplorationSvg from './svg/exploration.svg';
import LinkSvg from './svg/link.svg';
import ExternalLinkSvg from './svg/external-link.svg';
import ConnectionSvg from './svg/connection.svg';
import CertificationSvg from './svg/certification.svg';
import CloseSvg from './svg/close.svg';
import PlusSvg from './svg/plus.svg';
import DownloadSvg from './svg/download.svg';
import UploadSvg from './svg/upload.svg';

export enum IconName {
	Close = 'close',
	Plus = 'plus',
	Menu = 'menu',
	DotsVertical = 'dots-vertical',
	Pencil = 'pencil',
	Trash = 'trash',
	Send = 'send',
	Stop = 'stop',
	Check = 'check',
	Copy = 'copy',
	ChatBubble = 'chat-bubble',
	ChevronRight = 'chevron-right',
	NvidiaLogo = 'nvidia-logo',
	Database = 'database',
	Settings = 'settings',
	ChartBar = 'chart-bar',
	ChartLine = 'chart-line',
	Table = 'table',
	View = 'view',
	MaterializedView = 'materialized-view',
	Schema = 'schema',
	Column = 'column',
	Users = 'users',
	Key = 'key',
	Terms = 'terms',
	Exploration = 'exploration',
	Link = 'link',
	ExternalLink = 'external-link',
	Connection = 'connection',
	Certification = 'certification',
	Download = 'download',
	Upload = 'upload',
}

const icons: Record<IconName, FC<SVGProps<SVGSVGElement>>> = {
	[IconName.Close]: CloseSvg,
	[IconName.Plus]: PlusSvg,
	[IconName.Menu]: MenuSvg,
	[IconName.DotsVertical]: DotsVerticalSvg,
	[IconName.Pencil]: PencilSvg,
	[IconName.Trash]: TrashSvg,
	[IconName.Send]: SendSvg,
	[IconName.Stop]: StopSvg,
	[IconName.Check]: CheckSvg,
	[IconName.Copy]: CopySvg,
	[IconName.ChatBubble]: ChatBubbleSvg,
	[IconName.ChevronRight]: ChevronRightSvg,
	[IconName.NvidiaLogo]: NvidiaLogoSvg,
	[IconName.Database]: DatabaseSvg,
	[IconName.Settings]: SettingsSvg,
	[IconName.ChartBar]: ChartBarSvg,
	[IconName.ChartLine]: ChartLineSvg,
	[IconName.Table]: TableSvg,
	[IconName.View]: ViewSvg,
	[IconName.MaterializedView]: MaterializedViewSvg,
	[IconName.Schema]: SchemaSvg,
	[IconName.Column]: ColumnSvg,
	[IconName.Users]: UsersSvg,
	[IconName.Key]: KeySvg,
	[IconName.Terms]: TermsSvg,
	[IconName.Exploration]: ExplorationSvg,
	[IconName.Link]: LinkSvg,
	[IconName.ExternalLink]: ExternalLinkSvg,
	[IconName.Connection]: ConnectionSvg,
	[IconName.Certification]: CertificationSvg,
	[IconName.Download]: DownloadSvg,
	[IconName.Upload]: UploadSvg,
};

export type IconProps = SVGProps<SVGSVGElement> & {
	name: IconName;
};

export const Icon: FC<IconProps> = ({ name, ...props }) => {
	const SvgComponent = icons[name];
	return <SvgComponent {...props} />;
};
