// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Desktop MCP clients (Cursor, the Python SDK) register a short-lived loopback
// HTTP listener as the OAuth redirect. Navigating Chrome to that URL is what
// produces "This site can't be reached": the listener is gone, or bound to a
// different localhost address than Chrome resolved. These helpers decide when
// a Location is that listener and deliver the code with fetch instead of a
// top-level navigation.
//
// Only loopback HTTP(S) qualifies. A private-use scheme such as `cursor://` is
// left as an ordinary 302: the browser hands it to the OS protocol handler, and
// that navigation carries the user's consent click as its activation. Routing
// it through the handoff page would lose that activation and would let anyone
// who can craft a `/oauth/handoff?url=` link launch an arbitrary protocol
// handler from a GSF origin.

const isLoopbackHostname = (hostname: string): boolean =>
	hostname === 'localhost' ||
	hostname === '127.0.0.1' ||
	hostname === '::1' ||
	hostname === '[::1]';

const appOrigin = (requestUrl: string): string | null => {
	const appUrl = process.env.APP_URL;
	if (appUrl) {
		try {
			return new URL(appUrl).origin;
		} catch {
			// Fall through to the incoming request.
		}
	}
	try {
		return new URL(requestUrl).origin;
	} catch {
		return null;
	}
};

/** Loopback HTTP(S) only; never a web origin, a file, or a custom scheme. */
export const isLoopbackCallback = (url: URL): boolean => {
	if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;
	return isLoopbackHostname(url.hostname);
};

/** True when `url` is an MCP client's loopback listener, not a GSF page. */
export const isMcpClientCallback = (url: URL, requestUrl: string): boolean => {
	if (!isLoopbackCallback(url)) return false;
	const origin = appOrigin(requestUrl);
	// GSF pages are also loopback in local dev (localhost:3000). Anything else
	// on loopback — Cursor's :8787, Claude's ephemeral port — is the client.
	if (!origin) return true;
	return url.origin !== origin;
};

/** Absolute GSF `/oauth/handoff` URL that delivers `callback` without leaving GSF. */
export const mcpHandoffLocation = (callback: URL, requestUrl: string): string => {
	const handoff = new URL('/oauth/handoff', process.env.APP_URL || requestUrl);
	handoff.searchParams.set('url', callback.toString());
	return handoff.toString();
};

/** Same-origin path so the browser stays on this GSF host. */
export const mcpHandoffPath = (callback: URL): string => {
	const handoff = new URL('/oauth/handoff', 'http://gsf.invalid');
	handoff.searchParams.set('url', callback.toString());
	return `${handoff.pathname}${handoff.search}`;
};

const loopbackCandidates = (parsed: URL): string[] => {
	// `URL.hostname` accepts IPv6 only in bracketed form; a bare `::1` is
	// silently ignored and the previous hostname is kept.
	const aliases = ['localhost', '127.0.0.1', '[::1]'];
	return [
		parsed.toString(),
		...aliases
			.filter(
				(hostname) => parsed.hostname !== hostname && `[${parsed.hostname}]` !== hostname,
			)
			.map((hostname) => {
				const next = new URL(parsed);
				next.hostname = hostname;
				return next.toString();
			}),
	];
};

// Fetch from a public HTTPS origin (Astra) to localhost is often blocked by
// private-network access rules. A hidden iframe is a nested navigation, which
// Chrome still allows, so the listener can receive the code while this tab
// stays on GSF. The iframe is removed after a delay; its document is the
// client's response or Chrome's error page, neither of which the user should
// see.
const pokeLoopbackViaIframe = (url: string): void => {
	if (typeof document === 'undefined') return;
	const iframe = document.createElement('iframe');
	iframe.setAttribute('aria-hidden', 'true');
	iframe.title = 'oauth-callback';
	iframe.referrerPolicy = 'no-referrer';
	iframe.style.display = 'none';
	iframe.src = url;
	document.body.appendChild(iframe);
	window.setTimeout(() => iframe.remove(), 10_000);
};

// Fetch every localhost alias so an IPv4 Chrome tab still hits a listener that
// bound `[::1]:8787` (and the reverse). `no-cors` is required: the listener is
// not GSF and will not send ACAO. A live listener answers with an opaque
// response, which counts as fulfilled; a closed port rejects. Returns true only
// when at least one alias was reached, so callers can tell delivery from a dead
// listener. Each attempt has a deadline: a listener that accepts the socket
// but never answers would otherwise leave the page in its waiting state forever.
const DELIVERY_TIMEOUT_MS = 5_000;

export const deliverLoopbackCallback = async (redirectUrl: string): Promise<boolean> => {
	let parsed: URL;
	try {
		parsed = new URL(redirectUrl);
	} catch {
		return false;
	}
	if (!isLoopbackCallback(parsed)) return false;

	const results = await Promise.allSettled(
		loopbackCandidates(parsed).map((url) =>
			fetch(url, {
				mode: 'no-cors',
				credentials: 'omit',
				keepalive: true,
				signal: AbortSignal.timeout(DELIVERY_TIMEOUT_MS),
			}),
		),
	);
	const delivered = results.some((result) => result.status === 'fulfilled');
	if (!delivered) pokeLoopbackViaIframe(parsed.toString());
	return delivered;
};

const redirectUrlFromPayload = (data: unknown): string | null => {
	if (typeof data !== 'object' || data === null) return null;
	const payload = data as { redirect?: unknown; url?: unknown };
	if (payload.redirect !== true || typeof payload.url !== 'string') return null;
	return payload.url;
};

/** Keep the tab on GSF for a loopback callback; otherwise follow `redirectUrl`. */
export const followOAuthRedirect = (redirectUrl: string): void => {
	let parsed: URL;
	try {
		parsed = new URL(redirectUrl, window.location.origin);
	} catch {
		window.location.assign(redirectUrl);
		return;
	}
	if (isLoopbackCallback(parsed) && parsed.origin !== window.location.origin) {
		window.location.assign(mcpHandoffPath(parsed));
		return;
	}
	window.location.assign(parsed.toString());
};

// After SSO or password sign-in the login page has to resume `/oauth2/authorize`.
// A top-level navigation follows Better Auth's 302 to :8787. Asking for JSON
// (or reading a 302 Location) lets us send loopback callbacks to `/oauth/handoff`
// instead, so Chrome never paints "This site can't be reached".
export const resumeAuthorizeFromApp = async (destination: string): Promise<void> => {
	try {
		const response = await fetch(destination, {
			headers: { Accept: 'application/json' },
			credentials: 'include',
			redirect: 'manual',
		});
		if (response.status >= 300 && response.status < 400) {
			const location = response.headers.get('location');
			if (location) {
				followOAuthRedirect(location);
				return;
			}
		}
		if (response.ok) {
			const url = redirectUrlFromPayload(await response.json());
			if (url) {
				followOAuthRedirect(url);
				return;
			}
		}
	} catch {
		// Fall through to a real navigation; the authorize 302 rewrite is the
		// backup so login still completes if this fetch fails.
	}
	window.location.assign(destination);
};
