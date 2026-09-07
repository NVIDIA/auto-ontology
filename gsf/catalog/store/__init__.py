# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The catalog's persistence layer.

Everything above this package is storage-agnostic — extraction, parsing, the
diff. Keep it that way: SQL and storage types belong here and nowhere above.
"""
