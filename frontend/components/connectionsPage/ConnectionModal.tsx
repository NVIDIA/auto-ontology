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
import {
	CONNECTION_EITHER_FIELDS,
	connectionFieldsFor,
	ConnectionType,
	type ConnectionFieldKey,
} from '@/enums/connection';
import type { Connection, ConnectionInput, ConnectionParams } from '@/types/connection';

const TYPE_STEP = 'Select Connector';
const CONNECT_STEP = 'Connect';
const SCHEMA_STEP = 'Select Schemas';

/**
 * Connectors whose backend honours a `schemas` ingestion allowlist and returns a
 * schema list from the connection test. Types absent here skip the picker step
 * and ingest everything the credentials can see.
 */
const SCHEMA_SELECTION_TYPES: ReadonlySet<ConnectionType> = new Set([
	ConnectionType.DATABRICKS,
	ConnectionType.SNOWFLAKE,
	ConnectionType.KYUUBI,
	ConnectionType.TRINO,
]);

type FieldValues = Partial<Record<ConnectionFieldKey, string>>;

/** The form stores every value as a string; a checkbox is 'true'/''. */
const toFieldValue = (raw: unknown): string => {
	if (typeof raw === 'boolean') return raw ? 'true' : '';
	return typeof raw === 'string' ? raw : '';
};

const toFieldValues = (params: ConnectionParams): FieldValues => {
	const stored = params as Record<string, unknown>;
	return Object.fromEntries(
		connectionFieldsFor(params.type)
			.filter((field) => stored[field.key] != null)
			.map((field) => [field.key, toFieldValue(stored[field.key])]),
	);
};

export type ConnectionModalProps = {
	open: boolean;
	/**
	 * Editing when set. The connector type and database name are the connection's
	 * identity on the backend, so both are fixed for the life of the record.
	 */
	connection?: Connection | null;
	onConfirm: () => void;
	onCancel: () => void;
};

export const ConnectionModal = ({
	open,
	connection = null,
	onConfirm,
	onCancel,
}: ConnectionModalProps) => {
	const editing = connection != null;

	const [loading, setLoading] = useState(false);
	const [testingConnection, setTestingConnection] = useState(false);
	const [isConnectionTested, setIsConnectionTested] = useState(false);
	const [testSuccessMessage, setTestSuccessMessage] = useState<string | null>(null);
	const [activeStep, setActiveStep] = useState(0);
	const [connectionType, setConnectionType] = useState<ConnectionType>(
		connection?.connection.type ?? ConnectionType.POSTGRESQL,
	);
	const [values, setValues] = useState<FieldValues>(() =>
		connection == null ? {} : toFieldValues(connection.connection),
	);
	const [alert, setAlert] = useState<string | null>(null);

	// The stored allowlist is what the picker reopens on, and what a re-test
	// falls back to rather than silently dropping the connection's scope.
	const initialSchemas = useMemo(
		() => (connection?.connection as { schemas?: string[] } | undefined)?.schemas ?? [],
		[connection],
	);

	// `availableSchemas` is populated from the connection-test response, so
	// there's no separate fetch.
	const [availableSchemas, setAvailableSchemas] = useState<string[]>([]);
	const [selectedSchemas, setSelectedSchemas] = useState<string[]>(initialSchemas);

	/**
	 * Credentials are stripped from everything the API hands back, so an edit
	 * form cannot show them. Blank therefore means "keep the stored one", which
	 * is how the backend reads it too.
	 */
	const keepCurrentKeys = useMemo<ReadonlySet<ConnectionFieldKey>>(() => {
		if (connection == null) return new Set();
		const stored = connection.connection as Record<string, unknown>;
		return new Set(
			connectionFieldsFor(connection.connection.type)
				.filter((field) => field.secret === true && stored[field.key] == null)
				.map((field) => field.key),
		);
	}, [connection]);

	const readOnlyKeys = useMemo<ReadonlySet<ConnectionFieldKey>>(
		() => (editing ? new Set<ConnectionFieldKey>(['database']) : new Set()),
		[editing],
	);

	const supportsSchemaSelection = SCHEMA_SELECTION_TYPES.has(connectionType);

	// Naming a schema on the form replaces picking one from a list: the test has
	// already confirmed it exists, so there is nothing left to choose.
	const explicitSchema = (values.schema ?? '').trim();

	const steps = useMemo(() => {
		const base = editing ? [CONNECT_STEP] : [TYPE_STEP, CONNECT_STEP];
		return supportsSchemaSelection && !explicitSchema ? [...base, SCHEMA_STEP] : base;
	}, [editing, supportsSchemaSelection, explicitSchema]);

	// `testOnly` fields (the Databricks schema filter) shape the connection test
	// but must not end up on the stored connection, so they are dropped unless
	// the payload is headed for the test endpoint.
	const buildConnection = useCallback(
		({ forTest = false }: { forTest?: boolean } = {}): ConnectionInput => {
			const fields = connectionFieldsFor(connectionType).filter(
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

	const fieldsComplete = useMemo(() => {
		const allRequiredPresent = connectionFieldsFor(connectionType).every(
			(field) =>
				field.optional ||
				keepCurrentKeys.has(field.key) ||
				(values[field.key] ?? '').trim().length > 0,
		);
		// Alternative credentials are each optional on their own, so the "every
		// required field" check above cannot see that one of them is still needed.
		const alternatives = CONNECTION_EITHER_FIELDS[connectionType];
		const alternativeSatisfied =
			alternatives == null ||
			alternatives.some(
				(key) => keepCurrentKeys.has(key) || (values[key] ?? '').trim().length > 0,
			);

		return allRequiredPresent && alternativeSatisfied;
	}, [connectionType, values, keepCurrentKeys]);

	const onTypeStep = steps[activeStep] === TYPE_STEP;
	const canContinue = onTypeStep ? false : fieldsComplete;

	const handleNext = useCallback((): void => {
		setAlert(null);
		setActiveStep((prev) => Math.min(prev + 1, steps.length - 1));
	}, [steps.length]);

	const handleBack = useCallback((): void => {
		setAlert(null);
		setActiveStep((prev) => Math.max(prev - 1, 0));
	}, []);

	const resetSchemaState = useCallback((): void => {
		setAvailableSchemas([]);
		setSelectedSchemas(initialSchemas);
	}, [initialSchemas]);

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

		const res = await connectionsApi.test(
			buildConnection({ forTest: true }),
			connection?.database_name,
		);
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
	}, [buildConnection, connection]);

	const handleSubmit = useCallback(async (): Promise<void> => {
		if (!isConnectionTested) {
			setAlert(`Connection must be tested before ${editing ? 'saving' : 'creating'}.`);
			return;
		}

		setLoading(true);
		setAlert(null);
		const input = buildConnection();
		const res =
			connection == null
				? await connectionsApi.create(input)
				: await connectionsApi.update(connection.database_name, input);
		setLoading(false);

		if ('error' in res && res.error) {
			setAlert(res.message ?? `Failed to ${editing ? 'update' : 'create'} the connection.`);
			return;
		}

		onConfirm();
	}, [buildConnection, connection, editing, isConnectionTested, onConfirm]);

	const renderStepContent = (step: number) => {
		switch (steps[step]) {
			case TYPE_STEP:
				return <ConnectionTypeStep onSelect={handleSelectType} />;
			case CONNECT_STEP:
				return (
					<ConnectionConnectStep
						connectionType={connectionType}
						values={values}
						onFieldChange={handleFieldChange}
						readOnlyKeys={readOnlyKeys}
						keepCurrentKeys={keepCurrentKeys}
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
						<p className="px-2 text-xs text-secondary dark:text-zinc-400">
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
		if (onTypeStep) {
			return [{ label: 'Cancel', onClick: onCancel, variant: 'outline' }];
		}

		const actions: StepperFooterAction[] = [
			activeStep === 0
				? { label: 'Cancel', onClick: onCancel, variant: 'outline' }
				: { label: 'Back', onClick: handleBack, variant: 'outline' },
		];

		const isLastStep = activeStep === steps.length - 1;
		// Require a successful test before advancing so the schema list can load.
		const blocked = !canContinue || loading || testingConnection || !isConnectionTested;

		actions.push(
			isLastStep
				? {
						label: editing ? 'Save' : 'Create',
						onClick: () => {
							void handleSubmit();
						},
						disabled: blocked,
						loading,
					}
				: { label: 'Next', onClick: handleNext, disabled: blocked, loading },
		);

		return actions;
	}, [
		activeStep,
		canContinue,
		editing,
		handleBack,
		handleNext,
		handleSubmit,
		isConnectionTested,
		loading,
		onCancel,
		onTypeStep,
		steps.length,
		testingConnection,
	]);

	return (
		<ModalWithSteps
			open={open}
			onClose={onCancel}
			title={editing ? `Edit ${connection.database_name}` : 'Create New Connection'}
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
