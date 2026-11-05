# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch
import pytest
from auto_ontology.connectors.bigquery import BigQueryDatabase


@patch("auto_ontology.connectors.bigquery.bigquery.Client")
def test_bigquery_parameter_types(mock_client_cls):
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


@patch("auto_ontology.connectors.bigquery.bigquery.Client")
def test_bigquery_get_queries_hours(mock_client_cls):
    db = BigQueryDatabase(connection_url_or_project="proj", dataset_id="ds")
    df = db.get_queries(hours=12)
    assert list(df.columns) == ["end_time", "query_text"]