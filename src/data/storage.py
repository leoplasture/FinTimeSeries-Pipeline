"""SQLite storage layer for financial time series using SQLAlchemy."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


class DataStorage:
    """Persist and query financial time series with metadata versioning.

    Parameters:
       db_path (str): SQLite database path.

    Example:
       >>> storage = DataStorage(db_path="data/finance.db")
       >>> storage.store_dataframe(df, "aapl_daily", if_exists="replace")
       >>> out = storage.query_table("aapl_daily", conditions={"Volume": "> 1000000"})

    SQL example:
       SELECT Date, Close
       FROM aapl_daily
       WHERE Date BETWEEN '2024-01-01' AND '2024-06-30'
       ORDER BY Date;
    """

    def __init__(self, db_path: str = "data/finance.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # QueuePool settings are supported by SQLAlchemy for sqlite URLs.
        self.engine: Engine = create_engine(
            f"sqlite:///{self.db_path}",
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            future=True,
        )
        self.metadata_table = "_dataset_metadata"
        self._ensure_metadata_table()

    def store_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
        source: str = "unknown",
        cleaning_log: str = "{}",
        version: str = "v1",
        fetch_time: Optional[str] = None,
    ) -> None:
        """Store a DataFrame into SQLite and write dataset version metadata.

        Parameters:
           df (pandas.DataFrame): DataFrame to store.
           table_name (str): Target SQL table.
           if_exists (str): Pandas to_sql mode: "replace", "append", or "fail".
           source (str): Source provider name (for metadata).
           cleaning_log (str): Serialized cleaning report.
           version (str): Dataset version label.
           fetch_time (str | None): ISO timestamp. Uses UTC now when None.

        Raises:
           ValueError: If required columns are missing.
           RuntimeError: If database write operation fails.
        """
        required = {"Date", "Open", "High", "Low", "Close", "Volume", "Adj_Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        payload = df.copy()
        payload["Date"] = pd.to_datetime(payload["Date"], errors="coerce")
        payload = payload.dropna(subset=["Date"]).sort_values("Date")

        if fetch_time is None:
            fetch_time = pd.Timestamp.utcnow().isoformat()

        try:
            payload.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            self._create_date_index(table_name)
            self._upsert_metadata(
                table_name=table_name,
                fetch_time=fetch_time,
                source=source,
                cleaning_log=cleaning_log,
                version=version,
            )
        except SQLAlchemyError as exc:
            raise RuntimeError(
                f"Failed to store DataFrame into '{table_name}'."
            ) from exc

    def query_table(
        self,
        table_name: str,
        conditions: Optional[Dict[str, str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """Query a stored table with optional conditions and date filter.

        Parameters:
           table_name (str): Table name to query.
           conditions (dict | None): Mapping of `column -> SQL predicate fragment`.
              Example: {"Volume": "> 1000000", "Close": "< 200"}
           date_range (tuple[str, str] | None): Date bounds as (start, end).

        Returns:
           pandas.DataFrame: Query results ordered by Date.

        SQL example:
           query_table("aapl_daily", conditions={"Volume": "> 1000000"},
                    date_range=("2024-01-01", "2024-12-31"))
        """
        clauses = []
        if conditions:
            for col, cond in conditions.items():
                clauses.append(f"{col} {cond}")
        if date_range:
            start_date, end_date = date_range
            clauses.append(f"Date BETWEEN '{start_date}' AND '{end_date}'")

        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)

        query = f"SELECT * FROM {table_name}{where_sql} ORDER BY Date"
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(text(query), conn)
        except SQLAlchemyError as exc:
            raise RuntimeError(f"Failed to query table '{table_name}'.") from exc

    def get_table_list(self) -> pd.DataFrame:
        """Return all database table names.

        Returns:
           pandas.DataFrame: Single-column DataFrame with table names.
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        return pd.DataFrame({"table_name": tables})

    def export_to_csv(self, table_name: str, output_path: str) -> Path:
        """Export a SQL table to a CSV file.

        Parameters:
           table_name (str): Source SQL table.
           output_path (str): Destination CSV path.

        Returns:
           pathlib.Path: Path to the written CSV file.
        """
        result = self.query_table(table_name)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        return out_path

    def backup_database(self, backup_path: str) -> Path:
        """Create a physical backup copy of the SQLite database file.

        Parameters:
           backup_path (str): Destination backup path.

        Returns:
           pathlib.Path: Path to the backup file.
        """
        target = Path(backup_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.engine.dispose()
        shutil.copy2(self.db_path, target)
        return target

    def _ensure_metadata_table(self) -> None:
        ddl = f"""
      CREATE TABLE IF NOT EXISTS {self.metadata_table} (
         table_name TEXT PRIMARY KEY,
         fetch_time TEXT NOT NULL,
         source TEXT NOT NULL,
         cleaning_log TEXT NOT NULL,
         version TEXT NOT NULL
      )
      """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))

    def _create_date_index(self, table_name: str) -> None:
        index_name = f"idx_{table_name}_date"
        ddl = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (Date)"
        with self.engine.begin() as conn:
            conn.execute(text(ddl))

    def _upsert_metadata(
        self,
        table_name: str,
        fetch_time: str,
        source: str,
        cleaning_log: str,
        version: str,
    ) -> None:
        stmt = text(
            f"""
         INSERT INTO {self.metadata_table} (table_name, fetch_time, source, cleaning_log, version)
         VALUES (:table_name, :fetch_time, :source, :cleaning_log, :version)
         ON CONFLICT(table_name) DO UPDATE SET
            fetch_time = excluded.fetch_time,
            source = excluded.source,
            cleaning_log = excluded.cleaning_log,
            version = excluded.version
         """
        )
        with self.engine.begin() as conn:
            conn.execute(
                stmt,
                {
                    "table_name": table_name,
                    "fetch_time": fetch_time,
                    "source": source,
                    "cleaning_log": cleaning_log,
                    "version": version,
                },
            )
