// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useMemo, useState } from 'react';
import { connectionsApi } from '@/api/connections';
import { ModalWithSteps, type StepperFooterAction } from '@/components/ModalWithSteps';
import { ConnectionConnectStep } from '@/components/connectionsPage/steps/ConnectionConnectStep';
import { ConnectionTypeStep } from '@/components/connectionsPage/steps/ConnectionTypeStep';
import { CONNECTION_FIELDS, ConnectionType, type ConnectionFieldKey } from '@/enums/connection';
import type { ConnectionInput } from '@/types/connection';

const NEW_CONNECTION_STEPS = ['Select Connector', 'Connect'] as const;

type FieldValues = Partial<Record<ConnectionFieldKey, string>>;

export type NewConnectionsModalProps = {
	open: boolean;
	onConfirm: () => void;
	onCancel: () => void;
};

export const NewConnectionsModal = ({ open, onConfirm, onCancel }: NewConnectionsModalProps) => {
	const [loading, setLoading] = useState(false);
	const [testingConnection, setTestingConnection] = useState(false);
	const [isConnectionTested, setIsConnectionTested] = useState(false);
	const [testSuccessMessage, setTestSuccessMessage] = useState<string | null>(null);
	const [activeStep, setActiveStep] = useState(0);
	const [connectionType, setConnectionType] = useState<ConnectionType>(ConnectionType.POSTGRESQL);
	const [values, setValues] = useState<FieldValues>({});
	const [alert, setAlert] = useState<string | null>(null);

	const buildConnection = useCallback((): ConnectionInput => {
		const fields = CONNECTION_FIELDS[connectionType];
		const entries = fields.map((field) => [field.key, (values[field.key] ?? '').trim()]);
		// The discriminated union maps 1:1 to the per-type field keys.
		return { type: connectionType, ...Object.fromEntries(entries) } as ConnectionInput;
	}, [connectionType, values]);

	const canContinue = useMemo(() => {
		if (activeStep === 0) return false;
		return CONNECTION_FIELDS[connectionType].every(
			(field) => field.optional || (values[field.key] ?? '').trim().length > 0,
		);
	}, [activeStep, connectionType, values]);

	const handleNext = useCallback((): void => {
		setAlert(null);
		setActiveStep((prev) => Math.min(prev + 1, NEW_CONNECTION_STEPS.length - 1));
	}, []);

	const handleBack = useCallback((): void => {
		setAlert(null);
		setActiveStep((prev) => Math.max(prev - 1, 0));
	}, []);

	const handleSelectType = (type: ConnectionType): void => {
		setConnectionType(type);
		setValues({});
		setAlert(null);
		setIsConnectionTested(false);
		setTestSuccessMessage(null);
		setActiveStep(1);
	};

	const handleFieldChange = (key: ConnectionFieldKey, value: string): void => {
		setValues((prev) => ({ ...prev, [key]: value }));
		setIsConnectionTested(false);
		setTestSuccessMessage(null);
		setAlert(null);
	};

	const handleTestConnection = useCallback(async (): Promise<void> => {
		setTestingConnection(true);
		setAlert(null);
		setTestSuccessMessage(null);

		const res = await connectionsApi.test(buildConnection());
		setTestingConnection(false);

		if ('error' in res && res.error) {
			setIsConnectionTested(false);
			setAlert(res.message ?? 'Connection test failed.');
			return;
		}

		if (!('success' in res)) {
			setIsConnectionTested(false);
			setAlert('Connection test failed.');
			return;
		}

		setIsConnectionTested(true);
		setTestSuccessMessage('Connection successful.');
		setAlert(null);
	}, [buildConnection]);

	const handleCreate = useCallback(async (): Promise<void> => {
		if (!isConnectionTested) {
			setAlert('Connection must be tested before creating.');
			return;
		}

		setLoading(true);
		setAlert(null);
		const res = await connectionsApi.create(buildConnection());
		setLoading(false);

		if ('error' in res && res.error) {
			setAlert(res.message ?? 'Failed to create connection.');
			return;
		}

		onConfirm();
	}, [buildConnection, isConnectionTested, onConfirm]);

	const renderStepContent = (step: number) => {
		switch (step) {
			case 0:
				return <ConnectionTypeStep onSelect={handleSelectType} />;
			case 1:
				return (
					<ConnectionConnectStep
						connectionType={connectionType}
						values={values}
						onFieldChange={handleFieldChange}
						testSuccessMessage={testSuccessMessage}
						onTestConnection={() => {
							void handleTestConnection();
						}}
						testDisabled={!canContinue || loading}
						testingConnection={testingConnection}
					/>
				);
			default:
				return null;
		}
	};

	const footerActions: StepperFooterAction[] = useMemo(() => {
		if (activeStep === 0) {
			return [{ label: 'Cancel', onClick: onCancel, variant: 'outline' }];
		}

		const actions: StepperFooterAction[] = [
			{ label: 'Back', onClick: handleBack, variant: 'outline' },
		];

		const isLastStep = activeStep === NEW_CONNECTION_STEPS.length - 1;
		if (isLastStep) {
			actions.push({
				label: 'Create',
				onClick: () => {
					void handleCreate();
				},
				disabled: !canContinue || loading || testingConnection || !isConnectionTested,
				loading,
			});
		} else {
			actions.push({
				label: 'Next',
				onClick: handleNext,
				disabled: !canContinue || loading,
				loading,
			});
		}

		return actions;
	}, [
		activeStep,
		canContinue,
		handleBack,
		handleCreate,
		handleNext,
		isConnectionTested,
		loading,
		onCancel,
		testingConnection,
	]);

	return (
		<ModalWithSteps
			open={open}
			onClose={onCancel}
			title="Create New Connection"
			steps={[...NEW_CONNECTION_STEPS]}
			activeStep={activeStep}
			onActiveStepChange={setActiveStep}
			footerActions={footerActions}
			alert={alert}
		>
			{renderStepContent(activeStep)}
		</ModalWithSteps>
	);
};
