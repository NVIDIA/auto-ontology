"""Render a compact markdown table of the PowerBI and Tableau connection
schemas. Runs inside the openmetadata/ingestion image where the pydantic
models are already installed."""

from metadata.generated.schema.entity.services.connections.dashboard import (
    powerBIConnection,
    tableauConnection,
)


def render(label: str, mod) -> None:
    cls = next(
        c
        for n, c in vars(mod).items()
        if n.endswith("Connection") and hasattr(c, "model_json_schema")
    )
    s = cls.model_json_schema()
    print(f"## {label} — `{cls.__name__}`")
    print()
    title = s.get("title") or cls.__name__
    desc = (s.get("description") or "").strip()[:300]
    print(f"**Title:** {title}")
    if desc:
        print(f"\n**Description:** {desc}")
    print()
    print("| Field | Type | Required | Description |")
    print("| --- | --- | --- | --- |")
    req = set(s.get("required", []))
    for k, v in s.get("properties", {}).items():
        if "type" in v:
            t = v["type"]
        elif "$ref" in v:
            t = "`" + v["$ref"].split("/")[-1] + "`"
        elif "anyOf" in v:
            parts = []
            for a in v["anyOf"]:
                if "type" in a:
                    parts.append(a["type"])
                elif "$ref" in a:
                    parts.append("`" + a["$ref"].split("/")[-1] + "`")
            t = " | ".join(parts)
        else:
            t = "?"
        d = (v.get("description") or "").replace("|", " ").replace("\n", " ").strip()
        if len(d) > 140:
            d = d[:140] + "…"
        print(f"| `{k}` | {t} | {'yes' if k in req else ''} | {d} |")
    print()


render("PowerBI", powerBIConnection)
render("Tableau", tableauConnection)
