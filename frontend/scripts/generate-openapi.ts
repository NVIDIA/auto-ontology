// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Generate the public OpenAPI spec for GSF.
 *
 * The Next.js route handlers are the real API edge — the Python services are
 * ClusterIP-only and reachable only through this proxy — so the spec is built
 * from `app/api/**\/route.ts`, not from FastAPI directly. File-based routing
 * makes the paths deterministic and `withPermission(...)`/`withPublic` makes
 * the auth requirement readable off the AST.
 *
 * Handlers that call `proxyToBackend` forward `incoming.pathname` verbatim, so
 * their path matches the FastAPI path 1:1 and their request/response schemas
 * are merged in from `docs/openapi/backend.json`. Handlers with custom logic have no
 * machine-readable schema yet; they are emitted with an empty schema and
 * counted in the run summary so the gap stays visible.
 *
 * Run with `pnpm openapi`. The output is committed and CI diffs it.
 */

import { execFileSync } from 'node:child_process';
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { z } from 'zod';
import type { ZodObject, ZodType } from 'zod';
import { Node, Project, SyntaxKind } from 'ts-morph';
import type { ObjectLiteralExpression, VariableStatement } from 'ts-morph';
import type { OpenApiResponse, OpenApiRoute } from '../types/openapi';

const HTTP_METHODS = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options'] as const;
type HttpMethod = (typeof HTTP_METHODS)[number];

type Json = Record<string, unknown>;

type Operation = {
	path: string;
	method: HttpMethod;
	/** `null` for `withPublic`, otherwise the required permission map. */
	permissions: Record<string, string[]> | null;
	proxied: boolean;
	/** Backend path this handler calls directly, for hand-rolled proxies. */
	upstreamPath?: string;
	description?: string;
	file: string;
};

const scriptDir = dirname(fileURLToPath(import.meta.url));
const frontendDir = resolve(scriptDir, '..');
const repoRoot = resolve(frontendDir, '..');
const apiDir = resolve(frontendDir, 'app/api');
const backendSpecPath = resolve(repoRoot, 'docs/openapi/backend.json');
const outputPath = resolve(repoRoot, 'docs/openapi/gsf-api.json');
const mcpSpecPath = resolve(repoRoot, 'mcp/gsf_mcp/gsf-api.json');

/** `app/api/terms/[term_id]/route.ts` → `/api/terms/{term_id}`. */
const routePath = (filePath: string): string => {
	const segments = relative(apiDir, dirname(filePath))
		.split(sep)
		.filter(Boolean)
		// Route groups `(name)` organise files without appearing in the URL.
		.filter((segment) => !segment.startsWith('('))
		.map((segment) => segment.replace(/^\[\.{3}(.+)\]$/, '{$1}').replace(/^\[(.+)\]$/, '{$1}'));
	return `/api${segments.length ? `/${segments.join('/')}` : ''}`;
};

/** `{ catalog: ['read'], zone: ['manage'] }` → the same, as data. */
const readPermissions = (literal: ObjectLiteralExpression): Record<string, string[]> => {
	const result: Record<string, string[]> = {};
	literal.getProperties().forEach((property) => {
		if (!Node.isPropertyAssignment(property)) return;
		const initializer = property.getInitializer();
		if (!Node.isArrayLiteralExpression(initializer)) return;
		result[property.getName().replace(/['"]/g, '')] = initializer
			.getElements()
			.map((element) => element.getText().replace(/['"]/g, ''));
	});
	return result;
};

/**
 * The nodes to search for a given export: the initializer itself, plus any
 * same-file function it names. `export const GET = withPermission(...)(handler)`
 * keeps the `proxyToBackend` call in `handler`, one hop away.
 */
const searchScope = (statement: VariableStatement): Node[] => {
	const initializer = statement.getDeclarations()[0]?.getInitializer();
	if (!initializer) return [];
	const scope: Node[] = [initializer];
	const sourceFile = statement.getSourceFile();
	initializer.getDescendantsOfKind(SyntaxKind.Identifier).forEach((identifier) => {
		const declaration =
			sourceFile.getFunction(identifier.getText()) ??
			sourceFile.getVariableDeclaration(identifier.getText());
		if (declaration) scope.push(declaration);
	});
	return scope;
};

const callsFunction = (scope: Node[], name: string): boolean =>
	scope.some((node) =>
		node
			.getDescendantsOfKind(SyntaxKind.CallExpression)
			.some((call) => call.getExpression().getText().endsWith(name)),
	);

/**
 * The backend path a hand-rolled proxy calls, e.g. the chat routes'
 * `` fetch(`${PYTHON_API_URL}/api/chat/cancel?...`) ``.
 *
 * These handlers relay the upstream *response* but build the upstream
 * *request* themselves (different query names, session-derived ids), so only
 * their responses can be merged.
 */
const upstreamFetchPath = (scope: Node[]): string | undefined => {
	const template = scope
		.flatMap((node) => node.getDescendantsOfKind(SyntaxKind.TemplateExpression))
		.map((node) => node.getText())
		.find((text) => text.includes('${PYTHON_API_URL}'));
	const path = template?.match(/\$\{PYTHON_API_URL\}([^`?]*)/)?.[1];
	return path?.startsWith('/') ? path : undefined;
};

const leadingComment = (statement: VariableStatement): string | undefined => {
	const text = statement
		.getLeadingCommentRanges()
		.map((range) => range.getText())
		.join('\n')
		.replace(/^\s*\/\/ ?/gm, '')
		.replace(/^\/\*+|\*+\/$/g, '')
		.replace(/^\s*\* ?/gm, '')
		.trim();
	// Skip the licence header that opens every file.
	return !text || text.includes('SPDX-') ? undefined : text;
};

/**
 * The HTTP methods a `export const …` statement contributes.
 *
 * Usually one per statement (`export const GET = …`), but Better Auth's
 * catch-all mounts several at once via a destructuring export:
 * `export const { GET, POST } = toNextJsHandler(auth)`. Reading only the first
 * declaration's name yields the binding-pattern text and silently drops the
 * whole file.
 */
const exportedMethods = (statement: VariableStatement): HttpMethod[] => {
	const isMethod = (name: string): name is HttpMethod =>
		HTTP_METHODS.includes(name.toLowerCase() as HttpMethod);

	return statement
		.getDeclarations()
		.flatMap((declaration) => {
			const nameNode = declaration.getNameNode();
			if (Node.isObjectBindingPattern(nameNode)) {
				return nameNode.getElements().map((element) => element.getName());
			}
			return [declaration.getName()];
		})
		.filter(isMethod)
		.map((name) => name.toLowerCase() as HttpMethod);
};

const collectOperations = (): Operation[] => {
	const project = new Project({ skipAddingFilesFromTsConfig: true });
	project.addSourceFilesAtPaths(`${apiDir}/**/route.ts`);

	const operations: Operation[] = [];
	project.getSourceFiles().forEach((sourceFile) => {
		const path = routePath(sourceFile.getFilePath());
		const file = relative(repoRoot, sourceFile.getFilePath());
		let found = 0;

		sourceFile.getVariableStatements().forEach((statement) => {
			if (!statement.isExported()) return;

			const scope = searchScope(statement);
			const permissionCall = scope
				.flatMap((node) => node.getDescendantsOfKind(SyntaxKind.CallExpression))
				.find((call) => call.getExpression().getText().endsWith('withPermission'));
			const permissionArg = permissionCall?.getArguments()[0];
			if (permissionCall && !Node.isObjectLiteralExpression(permissionArg)) {
				// Falling back to "public" here would document a gated route as
				// needing no auth, so refuse to guess.
				throw new Error(
					`${file}: withPermission(${permissionArg?.getText() ?? ''}) is not an inline ` +
						'object literal, so the required permission cannot be read.',
				);
			}

			exportedMethods(statement).forEach((method) => {
				found += 1;
				operations.push({
					path,
					method,
					permissions: permissionArg
						? readPermissions(permissionArg as ObjectLiteralExpression)
						: null,
					proxied: callsFunction(scope, 'proxyToBackend'),
					upstreamPath: upstreamFetchPath(scope),
					description: leadingComment(statement),
					file,
				});
			});
		});

		// A route file that yields nothing is almost always an export shape this
		// walker doesn't understand, and silently dropping it publishes a spec
		// that claims to cover the whole surface while missing part of it.
		if (found === 0) throw new Error(`${file}: no HTTP method exports were recognised.`);
	});

	return operations.sort(
		(a, b) => a.path.localeCompare(b.path) || a.method.localeCompare(b.method),
	);
};

/**
 * `/api/schemas/{db_id}` → `/api/schemas/{}`.
 *
 * Both sides name path parameters in snake_case, so paths already match
 * literally. Matching on the blanked form keeps that from being load-bearing:
 * a route renamed on one side still merges instead of silently losing its
 * schema. The frontend's names are the ones published.
 */
const paramAgnostic = (path: string): string => path.replace(/\{[^}]+\}/g, '{}');

/**
 * Fail when a structural match paired two routes that don't actually agree.
 *
 * Both sides name path parameters identically, so the backend's parameter
 * objects are published as-is. That only holds while the names really are the
 * same — a divergence here means either a rename on one side or a `{}`-shape
 * collision between different routes, and both should stop the build rather
 * than publish a parameter the URL doesn't have.
 */
const assertParametersAlign = (parameters: Json[], path: string, backendPath: string): void => {
	const names = [...path.matchAll(/\{([^}]+)\}/g)].map((match) => match[1]);
	const declared = parameters
		.filter((parameter) => parameter.in === 'path')
		.map((parameter) => String(parameter.name));
	const agrees =
		declared.length === names.length && declared.every((name, index) => name === names[index]);
	if (!agrees) {
		throw new Error(
			`Path parameter mismatch: ${path} declares (${names.join(', ')}) ` +
				`but ${backendPath} declares (${declared.join(', ')}). ` +
				'The two routes were matched structurally but do not agree.',
		);
	}
};

/** `/api/terms/{term_id}/sql-attributes` → `terms`, the grouping tag. */
const resourceOf = (path: string): string => path.split('/')[2] ?? 'root';

/**
 * A one-line title for the sidebar, or nothing.
 *
 * Route comments are prose and wrap freely, so the first *line* is often a
 * fragment ("…every chat user may trigger"). Taking the first sentence and
 * requiring it to be short keeps genuine titles like
 * `zonesApi.getAll — list zones (viewable by all).` and drops the rest, which
 * survives in full as the description. Without a summary a viewer falls back
 * to `METHOD /path`, which reads better than a truncated sentence.
 */
const titleOf = (comment: string | undefined): string | undefined => {
	if (!comment) return undefined;
	const [sentence] = comment.replace(/\s+/g, ' ').split(/(?<=\.)\s/);
	return sentence.length <= 80 ? sentence : undefined;
};

const AUTH_RESPONSES: Json = {
	401: { description: 'No authenticated caller could be resolved' },
	403: { description: 'The caller lacks the required permission' },
};

// Of the three credentials resolveUser() accepts, the API token is the only one
// a script can obtain for itself — a session cookie needs a browser login and an
// SSO id token is minted for another service. So it is the one documented here,
// and the one a generated client should be wired to send.
const SECURITY_SCHEMES: Json = {
	ApiToken: {
		type: 'apiKey',
		in: 'header',
		name: 'x-api-key',
		description:
			'A GSF API token (`gsf_…`), created under Account → API Tokens. The token ' +
			'carries its owner’s permissions. `Authorization: Bearer <token>` is accepted too.',
	},
	SessionCookie: {
		type: 'apiKey',
		in: 'cookie',
		name: 'better-auth.session_token',
		description:
			'A browser session from an interactive sign-in. Only the few operations that ' +
			'manage API tokens require this — a token may not mint or revoke tokens.',
	},
};

const UNDOCUMENTED_BODY: Json = {
	description: 'Success. This handler has custom logic; its schema is not yet declared.',
	content: { 'application/json': { schema: {} } },
};

/**
 * Load every `app/api/**\/openapi.ts` declaration, keyed by route path.
 *
 * These modules carry the shapes that cannot be merged from `backend.json`:
 * the Prisma-backed handlers have no FastAPI counterpart at all, and the chat
 * handlers build their own upstream request. See `types/openapi.ts` for the
 * contract. They import nothing but zod, so a plain Node script can import
 * them — `route.ts` itself cannot be imported (`next/headers`, `after`,
 * `getPrisma()`).
 */
const loadDeclarations = async (): Promise<Map<string, OpenApiRoute>> => {
	// ts-morph rather than a glob library: it is already a dependency here, and
	// `node:fs`'s own glob is not in the installed @types/node.
	const project = new Project({ skipAddingFilesFromTsConfig: true });
	project.addSourceFilesAtPaths(`${apiDir}/**/openapi.ts`);
	const files = project
		.getSourceFiles()
		.map((sourceFile) => sourceFile.getFilePath())
		.sort();
	const entries = await Promise.all(
		files.map(async (file) => {
			const declaration = (await import(pathToFileURL(file).href)) as {
				openapi?: OpenApiRoute;
			};
			if (!declaration.openapi)
				throw new Error(`${relative(repoRoot, file)} has no \`openapi\` export`);
			return [routePath(file), declaration.openapi] as const;
		}),
	);
	return new Map(entries);
};

/**
 * zod → JSON Schema. The spec is OpenAPI 3.1, i.e. draft 2020-12 compatible.
 *
 * `$schema` is dropped throughout: zod declares the dialect on every schema it
 * emits, which is meaningless inside an OpenAPI document (the document's own
 * `openapi` version fixes the dialect) and noise in generated clients.
 */
const toJsonSchema = (schema: ZodType, io: 'input' | 'output'): Json => {
	const strip = (value: unknown): unknown => {
		if (Array.isArray(value)) return value.map(strip);
		if (!value || typeof value !== 'object') return value;
		return Object.fromEntries(
			Object.entries(value as Json)
				.filter(([key]) => key !== '$schema')
				.map(([key, child]) => [key, strip(child)]),
		);
	};
	return strip(z.toJSONSchema(schema, { io, unrepresentable: 'any' })) as Json;
};

const declaredResponse = (response: OpenApiResponse): Json => {
	const body: Json = { description: response.description };
	if (response.schema) {
		body.content = {
			[response.contentType ?? 'application/json']: {
				schema: toJsonSchema(response.schema, 'output'),
			},
		};
	}
	return body;
};

/**
 * Declare every `{param}` in the path that nothing else already declared.
 *
 * Proxied routes inherit their path parameters from FastAPI, but the
 * frontend-only routes have no upstream to inherit from — and OpenAPI requires
 * a matching `in: path` parameter for every template expression, so omitting
 * them makes the document invalid rather than merely thin. Deriving them from
 * the path means they cannot go missing whatever the route's provenance.
 *
 * A sibling `openapi.ts` may describe them via `path` to add a description or
 * a tighter schema; anything already present wins over the derived default.
 */
const assertPathParametersDeclared = (paths: Record<string, Json>): void => {
	const broken = Object.entries(paths).flatMap(([path, methods]) => {
		const names = [...path.matchAll(/\{([^}]+)\}/g)].map((match) => match[1]);
		if (!names.length) return [];
		return Object.entries(methods as Record<string, Json>).flatMap(([method, operation]) => {
			const declared = ((operation.parameters ?? []) as Json[])
				.filter((parameter) => parameter.in === 'path')
				.map((parameter) => parameter.name);
			const gaps = names.filter((name) => !declared.includes(name));
			return gaps.length ? [`${method.toUpperCase()} ${path} (${gaps.join(', ')})`] : [];
		});
	});
	if (broken.length) {
		throw new Error(
			`Path templates with undeclared parameters, which OpenAPI rejects:\n  ${broken.join('\n  ')}`,
		);
	}
};

const ensurePathParameters = (operation: Json, path: string, described: Json[]): void => {
	const names = [...path.matchAll(/\{([^}]+)\}/g)].map((match) => match[1]);
	if (!names.length) return;

	const existing = (operation.parameters ?? []) as Json[];
	const declared = new Set(
		existing.filter((parameter) => parameter.in === 'path').map((parameter) => parameter.name),
	);
	const byName = new Map(described.map((parameter) => [parameter.name, parameter]));

	const added = names
		.filter((name) => !declared.has(name))
		.map((name) => ({
			schema: { type: 'string' },
			...(byName.get(name) ?? {}),
			name,
			in: 'path',
			required: true,
		}));

	if (added.length) operation.parameters = [...existing, ...added];
};

/** A zod object of query params → one OpenAPI parameter per property. */
const declaredQuery = (schema: ZodObject): Json[] => {
	const json = toJsonSchema(schema, 'input');
	const required = new Set((json.required as string[] | undefined) ?? []);
	return Object.entries((json.properties ?? {}) as Record<string, Json>).map(
		([name, property]) => {
			const { description, ...rest } = property;
			return {
				name,
				in: 'query',
				required: required.has(name),
				...(description ? { description } : {}),
				schema: rest,
			};
		},
	);
};

/** Every `#/components/schemas/X` name referenced anywhere inside `value`. */
const refsIn = (value: unknown, into: Set<string> = new Set()): Set<string> => {
	if (!value || typeof value !== 'object') return into;
	Object.entries(value as Json).forEach(([key, child]) => {
		if (key === '$ref' && typeof child === 'string') {
			const name = child.split('/').pop();
			if (child.startsWith('#/components/schemas/') && name) into.add(name);
		} else {
			refsIn(child, into);
		}
	});
	return into;
};

/**
 * The subset of the backend's component schemas the published spec actually
 * uses, closed transitively.
 *
 * `backend.json` describes the whole internal FastAPI surface, but the public
 * spec exposes fewer operations, so copying every component ships schemas no
 * operation references — noise for a reader and for a client generator.
 * Closing over nested `$ref`s matters: a referenced envelope names its item
 * model, which may name others in turn.
 */
const reachableSchemas = (paths: Record<string, Json>, all: Json): Json => {
	const wanted = refsIn(paths);
	const queue = [...wanted];
	while (queue.length) {
		const name = queue.pop() as string;
		refsIn(all[name]).forEach((nested) => {
			if (!wanted.has(nested)) {
				wanted.add(nested);
				queue.push(nested);
			}
		});
	}
	const missing = [...wanted].filter((name) => !(name in all));
	if (missing.length) {
		throw new Error(`Spec references schemas absent from backend.json: ${missing.join(', ')}`);
	}
	return Object.fromEntries(
		Object.entries(all)
			.filter(([name]) => wanted.has(name))
			.sort(([a], [b]) => a.localeCompare(b)),
	);
};

const build = async (): Promise<{
	spec: Json;
	operations: Operation[];
	undocumented: Operation[];
}> => {
	const declarations = await loadDeclarations();
	const backend = JSON.parse(readFileSync(backendSpecPath, 'utf8')) as {
		paths: Record<string, Record<string, Json>>;
		components?: { schemas?: Json };
	};
	const { version } = JSON.parse(readFileSync(resolve(frontendDir, 'package.json'), 'utf8')) as {
		version: string;
	};

	const backendByShape = new Map<string, { path: string; methods: Record<string, Json> }>(
		Object.entries(backend.paths).map(([path, methods]) => [
			paramAgnostic(path),
			{ path, methods },
		]),
	);

	const operations = collectOperations();
	const undocumented: Operation[] = [];
	const paths: Record<string, Json> = {};

	operations.forEach((operation) => {
		const match = operation.proxied
			? backendByShape.get(paramAgnostic(operation.path))
			: undefined;
		const upstream = match?.methods[operation.method] ?? null;
		const merged: Json = { ...(upstream ?? {}) };
		if (match && Array.isArray(merged.parameters)) {
			assertParametersAlign(merged.parameters as Json[], operation.path, match.path);
		}

		// Tag every operation, not just the ones with a FastAPI counterpart:
		// `tags` would otherwise be inherited from upstream, and the ~28
		// frontend-only and relay handlers have no upstream to inherit from, so
		// viewers would file them under a nameless default group.
		merged.tags = [resourceOf(operation.path)];

		// A single-sentence route comment *is* the whole description, so deriving
		// the summary from it would print the same line twice in the docs. Keep
		// the upstream (FastAPI) summary in that case, and fall back to the
		// derived title only where nothing was inherited.
		const title = titleOf(operation.description);
		const description = operation.description?.replace(/\s+/g, ' ').trim();
		if (title && (title !== description || !merged.summary)) merged.summary = title;
		if (operation.description) merged.description = operation.description;
		merged.operationId = `${operation.method}${operation.path.replace(/[^a-zA-Z0-9]+/g, '_')}`;

		// A hand-rolled proxy relays the upstream response verbatim, so its
		// responses are borrowed even though its request shape is its own.
		const relayed = operation.upstreamPath
			? backendByShape.get(paramAgnostic(operation.upstreamPath))?.methods[operation.method]
			: undefined;
		if (relayed) merged['x-gsf-upstream'] = operation.upstreamPath;

		// Hand-written declarations win over anything inherited: where both
		// exist, the handler's own shape is the one callers actually see.
		const declared = declarations.get(operation.path)?.[operation.method];
		if (declared?.body) {
			merged.requestBody = {
				required: declared.body.required ?? true,
				...(declared.body.description ? { description: declared.body.description } : {}),
				content: {
					[declared.body.contentType ?? 'application/json']: {
						schema: toJsonSchema(declared.body.schema, 'input'),
					},
				},
			};
		}
		if (declared?.query) {
			const existing = (merged.parameters ?? []) as Json[];
			const added = declaredQuery(declared.query);
			const names = new Set(added.map((parameter) => parameter.name));
			merged.parameters = [...existing.filter((p) => !names.has(p.name)), ...added];
		}
		ensurePathParameters(
			merged,
			operation.path,
			declared?.path ? declaredQuery(declared.path) : [],
		);

		const upstreamResponses = ((upstream ?? relayed)?.responses ?? {}) as Json;
		const responses: Json = { ...upstreamResponses };
		Object.entries(declared?.responses ?? {}).forEach(([code, response]) => {
			// `null` removes a status inherited from FastAPI that this handler
			// can never actually return.
			if (response === null) delete responses[code];
			else responses[code] = declaredResponse(response);
		});

		// A 2xx counts as documented when it carries a schema, or when it was
		// declared by hand as deliberately body-less (a 204 delete, say) —
		// "no body" is an answer, unlike the inherited empty-schema placeholder.
		const declaredCodes = new Set(Object.keys(declared?.responses ?? {}));
		const documented = Object.entries(responses).some(
			([code, body]) =>
				code.startsWith('2') &&
				(Object.keys(((body as Json).content ?? {}) as Json).length > 0 ||
					declaredCodes.has(code)),
		);
		// AUTH_RESPONSES first: it is the generic fallback text for 401/403, so a
		// route that documented what its own 403 actually means must override it,
		// not be overwritten by it.
		merged.responses = {
			...(operation.permissions ? AUTH_RESPONSES : {}),
			...(documented ? responses : { ...responses, 200: UNDOCUMENTED_BODY }),
		};
		if (!documented) undocumented.push(operation);

		// An empty array marks a route as deliberately unauthenticated, which is
		// what `withPublic` means — without it the global requirement would apply.
		// A `sessionOnly` route is gated but rejects tokens, so it advertises the
		// cookie instead: publishing `ApiToken` there would send a generated
		// client straight into a 403.
		if (!operation.permissions) merged.security = [];
		else merged.security = declared?.sessionOnly ? [{ SessionCookie: [] }] : [{ ApiToken: [] }];

		merged['x-gsf-source'] = operation.file;
		merged['x-gsf-permissions'] = operation.permissions ?? 'public';
		merged['x-gsf-proxied'] = operation.proxied;

		paths[operation.path] = { ...(paths[operation.path] ?? {}), [operation.method]: merged };
	});

	assertPathParametersDeclared(paths);

	const spec: Json = {
		openapi: '3.1.0',
		info: {
			title: 'GSF API',
			version,
			description:
				'Public HTTP surface of GSF, served by the Next.js route handlers. ' +
				'Generated by `pnpm openapi` — do not edit by hand.',
		},
		security: [{ ApiToken: [] }],
		paths: Object.fromEntries(Object.entries(paths).sort(([a], [b]) => a.localeCompare(b))),
		components: {
			schemas: reachableSchemas(paths, backend.components?.schemas ?? {}),
			securitySchemes: SECURITY_SCHEMES,
		},
	};

	return { spec, operations, undocumented };
};

const { spec, operations, undocumented } = await build();
writeFileSync(outputPath, `${JSON.stringify(spec, null, 2)}\n`);

// Prettier owns formatting for committed files, and CI runs `format:check`.
execFileSync('npx', ['prettier', '--write', '--log-level', 'silent', outputPath], {
	cwd: frontendDir,
});

// The MCP server builds its tool surface from this spec and has to run from an
// ordinary `pip`/`uvx` install, where there is no docs/ directory to read — so a
// copy ships inside the package. Copied after Prettier so the two are
// byte-identical; CI diffs both paths, and gsf-mcp's tests compare them.
copyFileSync(outputPath, mcpSpecPath);

const paths = Object.keys(spec.paths as Json).length;
console.log(`${relative(repoRoot, outputPath)}: ${paths} paths, ${operations.length} operations`);
console.log(`  ${operations.length - undocumented.length} with schemas merged from the backend`);
if (undocumented.length) {
	console.log(`  ${undocumented.length} without a response schema (custom handlers):`);
	undocumented.forEach((operation) => {
		console.log(`    ${operation.method.toUpperCase()} ${operation.path}`);
	});
}
