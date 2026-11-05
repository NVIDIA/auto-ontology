# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from google.api_core.exceptions import GoogleAPIError
from google.auth import default as google_auth_default
from google.oauth2 import service_account
from google.cloud import bigquery

from auto_ontology.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)


class BigQueryDatabase(SQLDatabase):
    """
    Google BigQuery connector for Auto-Ontology.
    Supports Service Account JSON keys and Application Default Credentials (ADC).
    Includes cost guardrails (max bytes billed) and dataset-scoped isolation.
    """

    engine_name: str = "bigquery"

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        location: str = "US",
        credentials_json: Optional[str] = None,
        credentials_file: Optional[str] = None,
        maximum_bytes_billed: Optional[int] = 10737418240,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self.maximum_bytes_billed = maximum_bytes_billed
        self._client: Optional[bigquery.Client] = None

        self._init_client(credentials_json, credentials_file)

    def _init_client(
        self,
        credentials_json: Optional[str],
        credentials_file: Optional[str]
    ) -> None:
        """Initialize BigQuery client using Service Account credentials or ADC."""
        creds = None
        if credentials_json:
            info = json.loads(credentials_json)
            creds = service_account.Credentials.from_service_account_info(info)
        elif credentials_file:
            creds = service_account.Credentials.from_service_account_file(credentials_file)
        else:
            creds, _ = google_auth_default()

        self._client = bigquery.Client(
            project=self.project_id,
            credentials=creds,
            location=self.location,
        )

    def test_connection(self) -> bool:
        """Verify project and dataset accessibility."""
        try:
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            self._client.get_dataset(dataset_ref)
            return True
        except GoogleAPIError as e:
            logger.error(f"Failed to connect to BigQuery dataset {self.dataset_id}: {e}")
            return False

    def get_table_names(self) -> List[str]:
        """Fetch all table names in the dataset."""
        query = f"""
        SELECT table_name
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.TABLES`
        """
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.maximum_bytes_billed
        )
        query_job = self._client.query(query, job_config=job_config)
        return [row["table_name"] for row in query_job.result()]

    def run_sql(
        self,
        command: str,
        fetch: str = "all"
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """Execute Standard BigQuery SQL with safety limits."""
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.maximum_bytes_billed
        )
        query_job = self._client.query(command, job_config=job_config)
        results = query_job.result()
        rows = [dict(row) for row in results]

        if fetch == "one":
            return rows[0] if rows else {}
        return rows