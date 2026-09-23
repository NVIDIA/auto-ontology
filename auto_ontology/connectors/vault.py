# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HashiCorp Vault access for UI-managed database connection secrets.

Connection credentials are stored in Vault keyed by catalog database name. This
module is the single place that reads/writes those secrets; callers should treat
a missing or unreachable Vault as "no secret" rather than an error.

This module performs no import-time side effects (it does not load ``.env``);
entrypoints are responsible for loading the environment before use.
"""

from __future__ import annotations

import json
import logging
import os
import hvac

logger = logging.getLogger(__name__)

REQUIRED_VAULT_ENV_VARS = (
    "VAULT_ADDR",
    "VAULT_NAMESPACE",
    "VAULT_ROLE_ID",
    "VAULT_SECRET_ID",
)


def is_vault_configured() -> bool:
    """Return True if the user provided all required Vault env vars.

    Warn on partial configuration (some but not all set), which is almost
    always a misconfiguration that would otherwise silently disable Vault.
    """
    missing = [var for var in REQUIRED_VAULT_ENV_VARS if not os.environ.get(var)]
    if missing and len(missing) < len(REQUIRED_VAULT_ENV_VARS):
        logger.warning(
            "Vault is partially configured; ignoring Vault and treating "
            "connections as unencrypted. Missing env vars: %s",
            ", ".join(missing),
        )
    return not missing


def get_client() -> hvac.Client:
    addr = os.environ["VAULT_ADDR"]
    auth_mount = os.environ.get("VAULT_AUTH_MOUNT", "approle/nvdcs/dc1")
    try:
        client = hvac.Client(
            url=addr,
            namespace=os.environ["VAULT_NAMESPACE"],
        )
        client.auth.approle.login(
            role_id=os.environ["VAULT_ROLE_ID"],
            secret_id=os.environ["VAULT_SECRET_ID"],
            mount_point=auth_mount,
        )
    except Exception:
        logger.exception(
            "Failed to authenticate to Vault at %s (auth_mount=%s)", addr, auth_mount
        )
        raise
    logger.debug("Authenticated to Vault at %s (auth_mount=%s)", addr, auth_mount)
    return client


def write_secret(database_name: str, secret: dict[str, str]) -> None:
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    try:
        client = get_client()
        client.secrets.kv.v1.create_or_update_secret(
            path=database_name,
            secret={"connection": json.dumps(secret)},
            mount_point=mount_point,
        )
    except Exception:
        logger.exception(
            "Failed to write Vault secret for database %r (mount=%s)",
            database_name,
            mount_point,
        )
        raise
    logger.info(
        "Wrote Vault secret for database %r (mount=%s)", database_name, mount_point
    )


def read_secret(database_name: str) -> dict[str, str] | str:
    if not is_vault_configured():
        return ""
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    try:
        client = get_client()
        response = client.secrets.kv.v1.read_secret(
            path=database_name,
            mount_point=mount_point,
        )
        return json.loads(response["data"]["connection"])
    except Exception:
        logger.warning(
            "Could not read Vault secret for database %r (mount=%s); "
            "treating as no secret",
            database_name,
            mount_point,
            exc_info=True,
        )
        return ""


def delete_secrets(database_name: str | None = None) -> None:
    """Delete a single secret if database_name is given, otherwise delete all."""
    if not is_vault_configured():
        return

    client = get_client()
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")

    def delete(path: str) -> None:
        try:
            client.secrets.kv.v1.delete_secret(path=path, mount_point=mount_point)
        except Exception:
            logger.exception(
                "Failed to delete Vault secret %r (mount=%s)", path, mount_point
            )
            raise
        logger.info("Deleted Vault secret %r (mount=%s)", path, mount_point)

    if database_name:
        delete(database_name)
        return

    try:
        # Use GET with ?list=true instead of the LIST HTTP verb, which the
        # Vault endpoint/proxy rejects ("Unsupported HTTP method").
        response = client.adapter.get(
            f"v1/{mount_point}",
            params={"list": "true"},
        )
        keys = response["data"]["keys"]
    except hvac.exceptions.InvalidPath:
        # Nothing stored under this mount.
        logger.info("No secrets found under mount %r.", mount_point)
        return
    except Exception:
        logger.exception("Failed to list Vault secrets under mount %r", mount_point)
        raise

    logger.info("Deleting %d Vault secret(s) under mount %r", len(keys), mount_point)
    for key in keys:
        delete(key)
