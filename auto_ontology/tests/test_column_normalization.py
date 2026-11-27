# SPDX-FileCopyrightText: Copyright (c) 2026 Yzzz88
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from auto_ontology.catalog.normalize import normalize_columns


def test_wide_column_ordinals_survive_normalization() -> None:
    columns = normalize_columns(pd.DataFrame({"ordinal_position": [32768, None]}))
    assert columns.ordinal_position.iloc[0] == 32768
    assert pd.isna(columns.ordinal_position.iloc[1])
