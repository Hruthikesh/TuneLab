from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Column:
    name: str
    original_name: str
    column_type: str
    is_primary_key: bool = False
    foreign_key_to: Optional[Tuple[str, str]] = None  # (target_table, target_column)


@dataclass
class Table:
    name: str
    original_name: str
    columns: Dict[str, Column] = field(default_factory=dict)


@dataclass
class DatabaseSchema:
    db_id: str
    tables: Dict[str, Table] = field(default_factory=dict)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)

    def to_ddl_text(self) -> str:
        """Serializes the schema to a clean, readable text representation for prompts."""
        lines = [f"Database: {self.db_id}"]

        for table_name, table in self.tables.items():
            col_strs = []

            for col_name, col in table.columns.items():
                col_def = f"{col.original_name} {col.column_type}"

                if col.is_primary_key:
                    col_def += " PRIMARY KEY"

                if col.foreign_key_to:
                    tgt_t, tgt_c = col.foreign_key_to
                    col_def += f" REFERENCES {tgt_t}({tgt_c})"

                col_strs.append(col_def)

            cols_joined = ", ".join(col_strs)
            lines.append(
                f"CREATE TABLE {table.original_name} ({cols_joined});"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Converts schema to serializable dictionary."""
        return {
            "db_id": self.db_id,
            "tables": {
                t_name: {
                    "original_name": t.original_name,
                    "columns": {
                        c_name: {
                            "original_name": c.original_name,
                            "type": c.column_type,
                            "is_primary_key": c.is_primary_key,
                            "foreign_key_to": c.foreign_key_to,
                        }
                        for c_name, c in t.columns.items()
                    },
                }
                for t_name, t in self.tables.items()
            },
            "foreign_keys": self.foreign_keys,
        }


def parse_spider_schema(raw_schema_entry: Dict[str, Any]) -> DatabaseSchema:
    """Parses a single database schema entry from Spider's tables.json."""
    db_id = raw_schema_entry.get("db_id")

    if not db_id:
        raise ValueError("Schema entry missing 'db_id' field.")

    table_names_orig = raw_schema_entry.get("table_names_original", [])
    table_names = raw_schema_entry.get("table_names", [])
    col_names_orig = raw_schema_entry.get("column_names_original", [])
    col_names = raw_schema_entry.get("column_names", [])
    col_types = raw_schema_entry.get("column_types", [])
    primary_keys = set(raw_schema_entry.get("primary_keys", []))
    raw_foreign_keys = raw_schema_entry.get("foreign_keys", [])

    if len(col_names_orig) != len(col_types):
        raise ValueError(
            f"Database '{db_id}': column count mismatch "
            f"(columns: {len(col_names_orig)}, types: {len(col_types)})"
        )

    # Initialize tables
    tables: Dict[str, Table] = {}

    for t_orig, t_norm in zip(table_names_orig, table_names):
        tables[t_orig] = Table(
            name=t_norm,
            original_name=t_orig,
        )

    # Map column index to (table_orig, col_orig)
    # for foreign-key resolution.
    col_idx_map: Dict[int, Tuple[str, str]] = {}

    for col_idx, (table_idx, c_orig) in enumerate(col_names_orig):
        if table_idx == -1:
            # Special wildcard token in Spider: [-1, "*"]
            continue

        if table_idx >= len(table_names_orig):
            raise ValueError(
                f"Database '{db_id}': table index {table_idx} out of range."
            )

        table_orig = table_names_orig[table_idx]

        c_norm = (
            col_names[col_idx][1]
            if col_idx < len(col_names)
            else c_orig
        )

        c_type = (
            col_types[col_idx].upper()
            if col_idx < len(col_types)
            else "TEXT"
        )

        is_pk = col_idx in primary_keys

        column = Column(
            name=c_norm,
            original_name=c_orig,
            column_type=c_type,
            is_primary_key=is_pk,
        )

        tables[table_orig].columns[c_orig] = column
        col_idx_map[col_idx] = (table_orig, c_orig)

    # Process foreign keys from official tables.json
    parsed_fks: List[Dict[str, str]] = []

    for fk_pair in raw_foreign_keys:
        if not (
            isinstance(fk_pair, (list, tuple))
            and len(fk_pair) == 2
        ):
            continue

        src_col_idx, tgt_col_idx = fk_pair

        if (
            src_col_idx in col_idx_map
            and tgt_col_idx in col_idx_map
        ):
            src_table, src_col = col_idx_map[src_col_idx]
            tgt_table, tgt_col = col_idx_map[tgt_col_idx]

            tables[src_table].columns[src_col].foreign_key_to = (
                tgt_table,
                tgt_col,
            )

            parsed_fks.append(
                {
                    "source_table": src_table,
                    "source_column": src_col,
                    "target_table": tgt_table,
                    "target_column": tgt_col,
                }
            )

    return DatabaseSchema(
        db_id=db_id,
        tables=tables,
        foreign_keys=parsed_fks,
    )


def validate_sqlite_database(
    sqlite_path: Path | str,
    schema: Optional[DatabaseSchema] = None,
) -> Dict[str, Any]:
    """
    Inspects a SQLite database file and verifies readability
    and schema consistency.

    SQLite internal tables such as sqlite_sequence are excluded
    from validation because Spider's tables.json may contain
    sqlite_* metadata that is not exposed as an ordinary
    application table by sqlite_master.
    """
    path = Path(sqlite_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"SQLite file not found: {path}"
        )

    # Connect read-only using URI.
    uri = f"file:{path.resolve()}?mode=ro"

    try:
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=5.0,
        )
    except sqlite3.OperationalError:
        # Fallback if URI mode fails on older SQLite versions.
        conn = sqlite3.connect(
            str(path.resolve()),
            timeout=5.0,
        )

    try:
        cursor = conn.cursor()

        # Ignore SQLite internal tables.
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%';
            """
        )

        db_tables = [row[0] for row in cursor.fetchall()]

        table_columns: Dict[str, List[str]] = {}

        for table_name in db_tables:
            cursor.execute(
                f"PRAGMA table_info({table_name});"
            )

            # PRAGMA table_info yields:
            # (cid, name, type, notnull, dflt_value, pk)
            cols = [row[1] for row in cursor.fetchall()]
            table_columns[table_name] = cols

        discrepancies: List[str] = []

        if schema:
            for s_tname, s_table in schema.tables.items():

                # Spider's tables.json can contain SQLite internal
                # metadata such as sqlite_sequence. These are not
                # ordinary application tables and are intentionally
                # excluded from db_tables above.
                if s_tname.lower().startswith("sqlite_"):
                    continue

                matched_t = next(
                    (
                        table
                        for table in db_tables
                        if table.lower() == s_tname.lower()
                    ),
                    None,
                )

                if not matched_t:
                    discrepancies.append(
                        f"Table '{s_tname}' from schema "
                        f"missing in SQLite database."
                    )
                    continue

                sqlite_cols_lower = [
                    column.lower()
                    for column in table_columns[matched_t]
                ]

                for s_col in s_table.columns.keys():
                    if s_col.lower() not in sqlite_cols_lower:
                        discrepancies.append(
                            f"Column '{s_tname}.{s_col}' from schema "
                            f"missing in SQLite database."
                        )

        return {
            "readable": True,
            "table_count": len(db_tables),
            "tables": db_tables,
            "table_columns": table_columns,
            "discrepancies": discrepancies,
            "valid": len(discrepancies) == 0,
        }

    finally:
        conn.close()