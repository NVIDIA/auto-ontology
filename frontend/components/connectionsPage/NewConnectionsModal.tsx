// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useMemo, useState } from 'react';
import { connectionsApi } from '@/api/connections';
import { ModalWithSteps, type StepperFooterAction } from '@/common/modal';
import { ConnectionConnectStep } from '@/components/connectionsPage/steps/ConnectionConnectStep';
import { ConnectionSelectDataStep } from '@/components/connectionsPage/steps/ConnectionSelectDataStep';
import { ConnectionTypeStep } from '@/components/connectionsPage/steps/ConnectionTypeStep';
import { CONNECTION_FIELDS, ConnectionType, type ConnectionFieldKey } from '@/enums/connection';
import type { ConnectionInput } from '@/types/connection';

const BASE_STEPS = ['Select Connector', 'Connect'] as const;
const SCHEMA_STEP = 'Select Schemas';

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

	// `availableSchemas` is populated from the connection-test response, so
	// there's no separate fetch.
	const [availableSchemas, setAvailableSchemas] = useState<string[]>([]);
	const [selectedSchemas, setSelectedSchemas] = useState<string[]>([]);

	const supportsSchemaSelection =
		connectionType === ConnectionType.DATABRICKS || connectionType === ConnectionType.SNOWFLAKE;

	// Naming a schema on the form replaces picking one from a list: the test has
	// already confirmed it exists, so there is nothing left to choose.
	const explicitSchema = (values.schema ?? '').trim();

	const steps = useMemo(
		() =>
			supportsSchemaSelection && !explicitSchema
				? [...BASE_STEPS, SCHEMA_STEP]
				: [...BASE_STEPS],
		[supportsSchemaSelection, explicitSchema],
	);

	// `testOnly` fields (the Databricks schema filter) shape the connection test
	// but must not end up on the stored connection, so they are dropped unless
	// the payload is headed for the test endpoint.
	const buildConnection = useCallback(
		({ forTest = false }: { forTest?: boolean } = {}): ConnectionInput => {
			const fields = CONNECTION_FIELDS[connectionType].filter(
				(field) => forTest || !field.testOnly,
			);
			const entries = fields.map((field) => [
				field.key,
				// Checkbox fields go over the wire as real booleans; the form
				// stores them as 'true'/'' like every other value.
				field.boolean ? values[field.key] === 'true' : (values[field.key] ?? '').trim(),
			]);
			const base = { type: connectionType, ...Object.fromEntries(entries) };
			if (supportsSchemaSelection) {
				// A named schema is the allowlist; otherwise use whatever was picked.
				if (explicitSchema) {
					return { ...base, schemas: [explicitSchema] } as ConnectionInput;
				}
				if (selectedSchemas.length > 0) {
					return { ...base, schemas: selectedSchemas } as ConnectionInput;
				}
			}
			return base as ConnectionInput;
		},
		[connectionType, values, supportsSchemaSelection, selectedSchemas, explicitSchema],
	);

	const fieldsComplete = useMemo(
		() =>
			CONNECTION_FIELDS[connectionType].every(
				(field) => field.optional || (values[field.key] ?? '').trim().length > 0,
			),
		[connectionType, values],
	);

	const canContinue = activeStep === 0 ? false : fieldsComplete;

	const handleNext = useCallback((): void => {
		setAlert(null);
		setActiveStep((prev) => Math.min(prev + 1, steps.length - 1));
	}, [steps.length]);

	const handleBack = useCallback((): void => {
		setAlert(null);
		setActiveStep((prev) => Math.max(prev - 1, 0));
	}, []);

	const resetSchemaState = (): void => {
		setAvailableSchemas([]);
		setSelectedSchemas([]);
	};

	const handleSelectType = (type: ConnectionType): void => {
		setConnectionType(type);
		setValues({});
		setAlert(null);
		setIsConnectionTested(false);
		setTestSuccessMessage(null);
		resetSchemaState();
		setActiveStep(1);
	};

	const handleFieldChange = (key: ConnectionFieldKey, value: string): void => {
		setValues((prev) => ({ ...prev, [key]: value }));
		setIsConnectionTested(false);
		setTestSuccessMessage(null);
		setAlert(null);
		// Credentials changed — any previously fetched schemas are now stale.
		resetSchemaState();
	};

	const handleTestConnection = useCallback(async (): Promise<void> => {
		setTestingConnection(true);
		setAlert(null);
		setTestSuccessMessage(null);

		const res = await connectionsApi.test(buildConnection({ forTest: true }));
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
		// The test response carries the connection's schemas — feed the picker.
		setAvailableSchemas(res.schemas);
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
		switch (steps[step]) {
			case 'Select Connector':
				return <ConnectionTypeStep onSelect={handleSelectType} />;
			case 'Connect':
				return (
					<ConnectionConnectStep
						connectionType={connectionType}
						values={values}
						onFieldChange={handleFieldChange}
						testSuccessMessage={testSuccessMessage}
						onTestConnection={() => {
							void handleTestConnection();
						}}
						testDisabled={!fieldsComplete || loading}
						testingConnection={testingConnection}
					/>
				);
			case SCHEMA_STEP:
				return (
					<div className="flex flex-col gap-2">
						<p className="px-2 text-xs text-zinc-500 dark:text-zinc-400">
							Leave empty to ingest all schemas.
						</p>
						<ConnectionSelectDataStep
							availableDatabases={availableSchemas}
							selectedDatabases={selectedSchemas}
							onSelectionChange={setSelectedSchemas}
						/>
					</div>
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

		const isLastStep = activeStep === steps.length - 1;
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
			// Require a successful test before advancing so the schema list can
			// load.
			actions.push({
				label: 'Next',
				onClick: handleNext,
				disabled: !canContinue || loading || testingConnection || !isConnectionTested,
				loading,
			});
		}

		return actions;
	}, [
		activeStep,
		steps.length,
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
			steps={steps}
			activeStep={activeStep}
			onActiveStepChange={setActiveStep}
			footerActions={footerActions}
			alert={alert}
		>
			{renderStepContent(activeStep)}
		</ModalWithSteps>
	);
};
