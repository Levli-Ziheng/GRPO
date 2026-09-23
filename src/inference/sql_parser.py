"""Strict single-query format; fenced SQL is recoverable but format-invalid."""
from __future__ import annotations
import re

def sql_code(sql: str) -> str:
    """Mask literals, quoted identifiers and comments, retaining SQL punctuation."""
    pattern = r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\x60(?:\x60\x60|[^\x60])*\x60|\[[^\]]*\]|--[^\n]*|/\*[\s\S]*?\*/"
    return re.sub(pattern, lambda m: " " * len(m.group()), sql)

def query_reason(sql: str) -> str | None:
    if not sql.strip():
        return "empty_sql"
    if "\x00" in sql:
        return "nul_byte"
    code = sql_code(sql).strip()
    if not re.match(r"^(SELECT|WITH)\b", code, re.I):
        return "not_select_or_with"
    if ";" in code.rstrip(";").rstrip() or code.count(";") > 1:
        return "multiple_statements"
    forbidden = re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|VACUUM|PRAGMA|REINDEX|ANALYZE)\b", code, re.I)
    return f"forbidden_keyword:{forbidden.group().upper()}" if forbidden else None

def parse_sql(raw_output: str) -> dict:
    text = raw_output.strip()
    fenced = re.fullmatch(r"```(?:sql|sqlite)?\s*\n?([\s\S]*?)\n?\s*```", text, re.I)
    candidate = fenced.group(1).strip() if fenced else text
    reason = query_reason(candidate)
    return {"raw_output": raw_output, "parsed_sql": candidate if reason is None else "",
            "format_valid": reason is None and fenced is None, "parse_error": reason}
