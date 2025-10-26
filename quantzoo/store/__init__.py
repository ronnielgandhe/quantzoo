"""Data persistence layer with DuckDB and Parquet storage."""

from .duck import DuckStore

__all__ = ["DuckStore"]