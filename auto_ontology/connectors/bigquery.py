# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from google.api_core.exceptions import GoogleAPIError
from google.auth import default as google_auth_default
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd

from auto_ontology.connectors.base import SQLDatabase

logger = logging.getLogger(__name__)


class BigQueryDatabase(SQLDatabase):
    """
    Google BigQuery connector for Auto-Ontology.
    Supports Service Account credentials and Application Default Credentials (ADC).
    """

    dialect: str = "bigquery"

    def __init__(
        self,
        connection_url_or_project: str,
        dataset_id: Optional[str] = None,
        location: str = "US",
        credentials_json: Optional[str] = None,
        credentials_file: Optional[str] = None,
        maximum_bytes_billed: Optional[int] = 10737418240,
    ) -> None:
        # Support both connection strings (bigquery://project_id/dataset_id) and explicit parameters
        if connection_url_or_project.startswith("bigquery://") or "://" in connection_url_or_project:
            parsed = urlparse(connection_url_or_project)
            self.project_id = parsed.netloc
            self.dataset_id = parsed.path.strip("/")
            params = parse_qs(parsed.query)
            self.location = params.get("location", [location])[0]
        else:
            self.project_id = connection_url_or_project
            self.dataset_id = dataset_id or ""
            self.location = location

        self.maximum_bytes_billed = maximum_bytes_billed
        self._client: Optional[bigquery.Client] = None
        self._init_client(credentials_json, credentials_file)

    def _init_client(self, credentials_json: Optional[str], credentials_file: Optional[str]) -> None:
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

    @property
    def database_name(self) -> str:
        return f"{self.project_id}.{self.dataset_id}"

    def test_connection(self) -> bool:
        try:
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            self._client.get_dataset(dataset_ref)
            return True
        except GoogleAPIError as e:
            logger.error(f"Failed to connect to BigQuery dataset {self.dataset_id}: {e}")
            return False

    def get_table_names(self) -> List[str]:
        query = f"""
        SELECT table_name 
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.TABLES`
        """
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=self.maximum_bytes_billed)
        query_job = self._client.query(query, job_config=job_config)
        return [row["table_name"] for row in query_job.result()]

    def execute(self, query: str) -> pd.DataFrame:
        """Executes query and returns result as pandas DataFrame."""
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=self.maximum_bytes_billed)
        query_job = self._client.query(query, job_config=job_config)
        return query_job.to_dataframe()

    def run_sql(
        self, 
        command: str, 
        fetch: str = "all"
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """Executes query with lazy materialization for single row fetches."""
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=self.maximum_bytes_billed)
        query_job = self._client.query(command, job_config=job_config)
        results = query_job.result()

        if fetch == "one":
            iterator = iter(results)
            first_row = next(iterator, None)
            return dict(first_row) if first_row else {}

        return [dict(row) for row in results]

    def close(self) -> None:
        if self._client:
            self._client.close()