<!--
SPDX-FileCopyrightText: Copyright (c) 2026, souljoy.
SPDX-License-Identifier: Apache-2.0
-->

# Apache Ossie model interchange

Use **Settings → Import / Export** or the admin API endpoints
`POST /api/model/import` and `POST /api/model/export`. Select **Ossie** to
export a semantic model as YAML or as a ZIP with companion ontology and
mapping documents. To import a semantic model from a ZIP, extract and upload
`semantic_model.yaml`.

## Converter compatibility

| Ossie spec | `apache/ossie-converters` revision | Auto Ontology interchange |
| --- | --- | --- |
| [0.3 draft](https://github.com/apache/ossie/tree/d27a6beab519c8d3bf08ad5fc901c293205134eb) | [`4661b0d`](https://github.com/apache/ossie-converters/tree/4661b0d93c0ff79d060feeb7c9a79821c3879065) | The NVIDIA converter exports `0.3.0.dev0` and accepts the 0.3 series. The same revision's dbt converter emits `0.3.0.dev0`. |
| [0.2 draft](https://github.com/apache/ossie/tree/01a5488d840deda309aac68cc0b312c195369b8d) | [`cb4b46b`](https://github.com/apache/ossie-converters/tree/cb4b46b97bcdb9f2685ef2f7023b9134a4426da2) | Older deployments using this pin export `0.2.0.dev0` and accept the 0.2 series. For a 0.2 target, use its matching dbt converter. To import an existing 0.2 file into a deployment using the 0.3 pin, convert it to native YAML first. |

The 0.3 integration tests exercise a dbt-generated model with aliased fields,
relationships, and a `DBT` nonportable extension. Auto Ontology imports the
mapped model semantics. dbt-specific metadata in the extension has no native
representation and is dropped. Auto Ontology's own extension preserves native
metadata such as term names and identifiers. Keep the original model if you
need vendor-specific metadata for another tool. The deployment needs a
connection for each database referenced in the file.

## Migrate an existing Ossie 0.2 file

Convert the original 0.2 semantic model to native Auto Ontology YAML using the
pinned legacy converter. This also applies to `semantic_model.yaml` extracted
from a legacy ZIP. The command runs the conversion offline after downloading
the pinned converter and its dependencies.

```bash
uvx --from 'git+https://github.com/apache/ossie-converters.git@cb4b46b97bcdb9f2685ef2f7023b9134a4426da2#subdirectory=nvidia' \
  ossie-nvidia-auto-ontology export -i model.ossie.yaml -o model.auto-ontology.yaml
```

If the input uses `schema.table` sources, pass `--database-name` with the
matching database name. Import `model.auto-ontology.yaml` using the usual UI or
API. You can then export the imported model as Ossie 0.3 if needed.
