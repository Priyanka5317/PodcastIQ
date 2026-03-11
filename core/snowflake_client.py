from typing import Any, Dict, List, Optional
import snowflake.connector
from core.config import SNOWFLAKE_CONFIG


class SnowflakeClient:
    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config or SNOWFLAKE_CONFIG
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = snowflake.connector.connect(
                account=self.config["account"],
                user=self.config["user"],
                password=self.config["password"],
                warehouse=self.config["warehouse"],
                database=self.config["database"],
                schema=self.config["schema"],
            )
        return self.conn

    def execute(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params or ())
            columns = [col[0] for col in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    def execute_scalar(self, sql: str, params: Optional[tuple] = None) -> Any:
        rows = self.execute(sql, params)
        if not rows:
            return None
        first_row = rows[0]
        if not first_row:
            return None
        return list(first_row.values())[0]

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
