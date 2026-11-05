# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union
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
    Fully implements SQLDatabase abstract interface contracts.
    """

    dialect: str = "bigquery"

    def __init__(
        self,
        connection_url_or_project: str,
        dataset_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials_json: Optional[str] = None,
        credentials_file: Optional[str] = None,
        maximum_bytes_billed: Optional[int] = 10737418240,
    ) -> None:
        if connection_url_or_project.startswith("bigquery://") or "://" in connection_url_or_project:
            parsed = urlparse(connection_url_or_project)
            self.project_id = parsed.netloc
            self.dataset_id = parsed.path.strip("/")
            params = parse_qs(parsed.query)
            loc_val = params.get("location", [None])[0]
            self.location = loc_val.strip() if loc_val and loc_val.strip() else location
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
            info = json.loads(credentials_json) if isinstance(credentials_json, str) else credentials_json
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

    def _get_job_config(self, parameters: Optional[list] = None) -> bigquery.QueryJobConfig:
        config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.maximum_bytes_billed,
            default_dataset=f"{self.project_id}.{self.dataset_id}" if self.dataset_id else None,
        )
        if parameters:
            query_params = []
            for p in parameters:
                if p is None:
                    query_params.append(bigquery.ScalarQueryParameter(None, "STRING", None))
                elif isinstance(p, bool):  # Check bool before int
                    query_params.append(bigquery.ScalarQueryParameter(None, "BOOL", p))
                elif isinstance(p, int):
                    query_params.append(bigquery.ScalarQueryParameter(None, "INT64", p))
                elif isinstance(p, float):
                    query_params.append(bigquery.ScalarQueryParameter(None, "FLOAT64", p))
                else:
                    query_params.append(bigquery.ScalarQueryParameter(None, "STRING", str(p)))
            config.query_parameters = query_params
        return config

    def test_connection(self) -> bool:
        try:
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            self._client.get_dataset(dataset_ref)
            return True
        except GoogleAPIError as e:
            logger.error(f"Failed to connect to BigQuery dataset {self.dataset_id}: {e}")
            return False

    def get_table_names(self) -> List[str]:
        query = f"SELECT table_name FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.TABLES`"
        query_job = self._client.query(query, job_config=self._get_job_config())
        return [row["table_name"] for row in query_job.result()]

    def get_tables(self) -> pd.DataFrame:
        query = f"""
        SELECT 
            table_schema,
            table_name,
            CASE 
                WHEN table_type = 'VIEW' THEN 'view'
                WHEN table_type = 'BASE TABLE' THEN 'table'
                ELSE LOWER(table_type)
            END AS table_type
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.TABLES`
        """
        return self.execute(query)

    def get_columns(self) -> pd.DataFrame:
        query = f"""
        SELECT 
            c.table_schema,
            c.table_name,
            c.column_name,
            c.data_type,
            (c.is_nullable = 'YES') AS is_nullable,
            COALESCE(p.description, '') AS description
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.COLUMNS` c
        LEFT JOIN `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS` p
            ON c.table_name = p.table_name AND c.column_name = p.column_name
        """
        return self.execute(query)

    def get_views(self) -> pd.DataFrame:
        query = f"""
        SELECT 
            table_schema,
            table_name,
            view_definition
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.VIEWS`
        """
        return self.execute(query)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_pks(self) -> pd.DataFrame:
        query = f"""
        SELECT 
            k.table_schema,
            k.table_name,
            k.column_name,
            k.ordinal_position
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` k
        JOIN `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` c
            ON k.constraint_name = c.constraint_name
        WHERE c.constraint_type = 'PRIMARY KEY'
        """
        try:
            return self.execute(query)
        except Exception:
            return pd.DataFrame(columns=["table_schema", "table_name", "column_name", "ordinal_position"])

    def get_fks(self) -> pd.DataFrame:
        query = f"""
        SELECT 
            k.table_schema,
            k.table_name,
            k.column_name,
            r.unique_constraint_schema AS referenced_schema,
            r.unique_constraint_name AS referenced_table,
            k.column_name AS referenced_column
        FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` k
        JOIN `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS` r
            ON k.constraint_name = r.constraint_name
        """
        try:
            return self.execute(query)
        except Exception:
            return pd.DataFrame(columns=["table_schema", "table_name", "column_name", "referenced_schema", "referenced_table", "referenced_column"])

    def execute(self, query: str, parameters: Optional[list] = None) -> pd.DataFrame:
        job_config = self._get_job_config(parameters)
        query_job = self._client.query(query, job_config=job_config)
        return query_job.to_dataframe()

    def run_sql(
        self, 
        command: str, 
        fetch: str = "all"
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        job_config = self._get_job_config()
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