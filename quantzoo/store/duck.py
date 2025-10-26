"""DuckDB-based data storage for QuantZoo artifacts."""

import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import duckdb
import subprocess


class Store(ABC):
    """Abstract base class for data storage."""
    
    @abstractmethod
    def write_trades(self, df: pd.DataFrame, run_id: str, metadata: Optional[Dict] = None) -> None:
        """Write trades data."""
        pass
    
    @abstractmethod
    def write_equity(self, df: pd.DataFrame, run_id: str, metadata: Optional[Dict] = None) -> None:
        """Write equity curve data."""
        pass
    
    @abstractmethod
    def read_trades(self, run_id: str) -> Optional[pd.DataFrame]:
        """Read trades data."""
        pass
    
    @abstractmethod
    def read_equity(self, run_id: str) -> Optional[pd.DataFrame]:
        """Read equity curve data."""
        pass
    
    @abstractmethod
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all stored runs."""
        pass


class DuckStore(Store):
    """DuckDB-based storage with Parquet files."""
    
    def __init__(self, db_path: str = "artifacts/quantzoo.duckdb", artifacts_dir: str = "artifacts"):
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        
        # Ensure artifacts directory exists
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize DuckDB database with required tables."""
        with duckdb.connect(self.db_path) as con:
            # Create metadata table for tracking runs
            con.execute("""
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMP,
                    seed INTEGER,
                    commit_sha VARCHAR,
                    strategy_name VARCHAR,
                    config_path VARCHAR,
                    trades_path VARCHAR,
                    equity_path VARCHAR,
                    total_trades INTEGER,
                    total_return DOUBLE,
                    sharpe_ratio DOUBLE,
                    max_drawdown DOUBLE
                )
            """)
            
            # Create trades table (virtual table backed by Parquet files)
            con.execute("""
                CREATE TABLE IF NOT EXISTS trades AS 
                SELECT * FROM read_parquet('artifacts/trades_*.parquet')
                WHERE 1=0
            """)
            
            # Create equity table (virtual table backed by Parquet files)
            con.execute("""
                CREATE TABLE IF NOT EXISTS equity AS 
                SELECT * FROM read_parquet('artifacts/equity_*.parquet')
                WHERE 1=0
            """)
    
    def _get_commit_sha(self) -> str:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, 
                text=True, 
                cwd=os.path.dirname(self.db_path)
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def write_trades(self, df: pd.DataFrame, run_id: str, metadata: Optional[Dict] = None) -> None:
        """Write trades data to Parquet and register in DuckDB."""
        if df.empty:
            return
        
        # Add run_id to dataframe
        df_copy = df.copy()
        df_copy['run_id'] = run_id
        
        # Write to Parquet
        trades_path = os.path.join(self.artifacts_dir, f"trades_{run_id}.parquet")
        df_copy.to_parquet(trades_path, index=False)
        
        # Register in DuckDB
        with duckdb.connect(self.db_path) as con:
            # Check if trades table exists, if not create it
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS trades AS 
                SELECT * FROM read_parquet('{trades_path}')
                WHERE 1=0
            """)
            
            # Insert new data
            con.execute(f"INSERT INTO trades SELECT * FROM read_parquet('{trades_path}')")
        
        # Update metadata
        if metadata:
            self._update_run_metadata(run_id, trades_path=trades_path, **metadata)
    
    def write_equity(self, df: pd.DataFrame, run_id: str, metadata: Optional[Dict] = None) -> None:
        """Write equity curve data to Parquet and register in DuckDB."""
        if df.empty:
            return
        
        # Add run_id to dataframe
        df_copy = df.copy()
        df_copy['run_id'] = run_id
        
        # Write to Parquet
        equity_path = os.path.join(self.artifacts_dir, f"equity_{run_id}.parquet")
        df_copy.to_parquet(equity_path, index=False)
        
        # Register in DuckDB
        with duckdb.connect(self.db_path) as con:
            # Check if equity table exists, if not create it
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS equity AS 
                SELECT * FROM read_parquet('{equity_path}')
                WHERE 1=0
            """)
            
            # Insert new data
            con.execute(f"INSERT INTO equity SELECT * FROM read_parquet('{equity_path}')")
        
        # Update metadata
        if metadata:
            self._update_run_metadata(run_id, equity_path=equity_path, **metadata)
    
    def read_trades(self, run_id: str) -> Optional[pd.DataFrame]:
        """Read trades data for a specific run."""
        trades_path = os.path.join(self.artifacts_dir, f"trades_{run_id}.parquet")
        
        if not os.path.exists(trades_path):
            return None
        
        return pd.read_parquet(trades_path)
    
    def read_equity(self, run_id: str) -> Optional[pd.DataFrame]:
        """Read equity curve data for a specific run."""
        equity_path = os.path.join(self.artifacts_dir, f"equity_{run_id}.parquet")
        
        if not os.path.exists(equity_path):
            return None
        
        return pd.read_parquet(equity_path)
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all stored runs with metadata."""
        with duckdb.connect(self.db_path) as con:
            try:
                result = con.execute("SELECT * FROM run_metadata ORDER BY timestamp DESC").fetchall()
                columns = [desc[0] for desc in con.description]
                
                runs = []
                for row in result:
                    run_dict = dict(zip(columns, row))
                    runs.append(run_dict)
                
                return runs
            except:
                return []
    
    def _update_run_metadata(
        self, 
        run_id: str, 
        seed: Optional[int] = None,
        strategy_name: Optional[str] = None,
        config_path: Optional[str] = None,
        trades_path: Optional[str] = None,
        equity_path: Optional[str] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """Update run metadata in database."""
        with duckdb.connect(self.db_path) as con:
            # Check if record exists
            existing = con.execute(
                "SELECT run_id FROM run_metadata WHERE run_id = ?", 
                [run_id]
            ).fetchone()
            
            timestamp = datetime.now()
            commit_sha = self._get_commit_sha()
            
            if existing:
                # Update existing record
                update_fields = []
                params = []
                
                if seed is not None:
                    update_fields.append("seed = ?")
                    params.append(seed)
                if strategy_name is not None:
                    update_fields.append("strategy_name = ?")
                    params.append(strategy_name)
                if config_path is not None:
                    update_fields.append("config_path = ?")
                    params.append(config_path)
                if trades_path is not None:
                    update_fields.append("trades_path = ?")
                    params.append(trades_path)
                if equity_path is not None:
                    update_fields.append("equity_path = ?")
                    params.append(equity_path)
                
                if metrics:
                    for metric_name in ['total_trades', 'total_return', 'sharpe_ratio', 'max_drawdown']:
                        if metric_name in metrics:
                            update_fields.append(f"{metric_name} = ?")
                            params.append(metrics[metric_name])
                
                if update_fields:
                    update_fields.append("timestamp = ?")
                    params.append(timestamp)
                    params.append(run_id)
                    
                    sql = f"UPDATE run_metadata SET {', '.join(update_fields)} WHERE run_id = ?"
                    con.execute(sql, params)
            else:
                # Insert new record
                con.execute("""
                    INSERT INTO run_metadata (
                        run_id, timestamp, seed, commit_sha, strategy_name, config_path,
                        trades_path, equity_path, total_trades, total_return, sharpe_ratio, max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id, timestamp, seed, commit_sha, strategy_name, config_path,
                    trades_path, equity_path,
                    metrics.get('total_trades', 0) if metrics else 0,
                    metrics.get('total_return', 0.0) if metrics else 0.0,
                    metrics.get('sharpe_ratio', 0.0) if metrics else 0.0,
                    metrics.get('max_drawdown', 0.0) if metrics else 0.0
                ])
    
    def query_trades(self, where_clause: str = "", limit: Optional[int] = None) -> pd.DataFrame:
        """Query trades with SQL-like syntax."""
        with duckdb.connect(self.db_path) as con:
            sql = "SELECT * FROM trades"
            if where_clause:
                sql += f" WHERE {where_clause}"
            if limit:
                sql += f" LIMIT {limit}"
            
            try:
                return con.execute(sql).df()
            except:
                return pd.DataFrame()
    
    def query_equity(self, where_clause: str = "", limit: Optional[int] = None) -> pd.DataFrame:
        """Query equity curves with SQL-like syntax."""
        with duckdb.connect(self.db_path) as con:
            sql = "SELECT * FROM equity"
            if where_clause:
                sql += f" WHERE {where_clause}"
            if limit:
                sql += f" LIMIT {limit}"
            
            try:
                return con.execute(sql).df()
            except:
                return pd.DataFrame()
    
    def get_run_summary(self) -> pd.DataFrame:
        """Get summary of all runs."""
        with duckdb.connect(self.db_path) as con:
            try:
                return con.execute("""
                    SELECT 
                        run_id,
                        timestamp,
                        strategy_name,
                        total_trades,
                        total_return,
                        sharpe_ratio,
                        max_drawdown
                    FROM run_metadata 
                    ORDER BY timestamp DESC
                """).df()
            except:
                return pd.DataFrame()
    
    def cleanup_old_runs(self, keep_last_n: int = 50) -> None:
        """Clean up old run data, keeping only the most recent N runs."""
        with duckdb.connect(self.db_path) as con:
            try:
                # Get old run IDs
                old_runs = con.execute(f"""
                    SELECT run_id, trades_path, equity_path 
                    FROM run_metadata 
                    ORDER BY timestamp DESC 
                    OFFSET {keep_last_n}
                """).fetchall()
                
                for run_id, trades_path, equity_path in old_runs:
                    # Delete Parquet files
                    if trades_path and os.path.exists(trades_path):
                        os.remove(trades_path)
                    if equity_path and os.path.exists(equity_path):
                        os.remove(equity_path)
                    
                    # Delete from database
                    con.execute("DELETE FROM trades WHERE run_id = ?", [run_id])
                    con.execute("DELETE FROM equity WHERE run_id = ?", [run_id])
                    con.execute("DELETE FROM run_metadata WHERE run_id = ?", [run_id])
                
                print(f"Cleaned up {len(old_runs)} old runs")
            except Exception as e:
                print(f"Error during cleanup: {e}")


def create_store(store_type: str = "duck", **kwargs) -> Store:
    """Factory function to create storage backends."""
    if store_type.lower() == "duck":
        db_path = kwargs.get('db_path', 'artifacts/quantzoo.duckdb')
        artifacts_dir = kwargs.get('artifacts_dir', 'artifacts')
        return DuckStore(db_path, artifacts_dir)
    else:
        raise ValueError(f"Unknown store type: {store_type}")