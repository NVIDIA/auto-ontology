# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from auto_ontology.connectors.bigquery import BigQueryDatabase
from auto_ontology.connectors.connection_string_factory import build_connection_string


@patch("auto_ontology.connectors.bigquery.google_auth_default")
@patch("auto_ontology.connectors.bigquery.bigquery.Client")
def test_bigquery_parameter_types(mock_client_cls: MagicMock, mock_auth: MagicMock) -> None:
    mock_auth.return_value = (MagicMock(), "test-project")
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    db = BigQueryDatabase(connection_url_or_project="proj", dataset_id="ds")
    config = db._get_job_config([True, 42, None, "test"])
    
    params = config.query_parameters
    assert params[0].type_ == "BOOL"
    assert params[0].value is True
    assert params[1].type_ == "INT64"
    assert params[1].value == 42
    assert params[2].value is None
    assert params[3].type_ == "STRING"


@patch("auto_ontology.connectors.bigquery.google_auth_default")
@patch("auto_ontology.connectors.bigquery.bigquery.Client")
def test_bigquery_get_queries_hours(mock_client_cls: MagicMock, mock_auth: MagicMock) -> None:
    mock_auth.return_value = (MagicMock(), "test-project")
    db = BigQueryDatabase(connection_url_or_project="proj", dataset_id="ds")
    df = db.get_queries(hours=12)
    assert list(df.columns) == ["end_time", "query_text"]


def test_build_connection_string_bigquery() -> None:
    # Test valid configuration with location
    conn = {"engine": "bigquery", "project": "my-project", "dataset": "my_ds", "location": "US"}
    assert build_connection_string(conn) == "bigquery://my-project/my_ds?location=US"

    # Test valid configuration without location
    conn_no_loc = {"engine": "bigquery", "project": "my-project", "dataset": "my_ds"}
    assert build_connection_string(conn_no_loc) == "bigquery://my-project/my_ds"

    # Test missing required project
    with pytest.raises(ValueError):
        build_connection_string({"engine": "bigquery", "dataset": "my_ds"})