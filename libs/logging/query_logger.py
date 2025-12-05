import json
import os
import sqlite3
from datetime import datetime, timezone
from threading import Lock
from typing import Any

# --------------------------------------------------------------------
# 目录初始化
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_DIR = os.path.join(BASE_DIR, "db")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "query.log")
DB_FILE = os.path.join(DB_DIR, "query_logs.db")


# --------------------------------------------------------------------
# SQLite 初始化（包含自动建表）
# --------------------------------------------------------------------
_sqlite_lock = Lock()


def _init_sqlite():
    with _sqlite_lock:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT,
                query TEXT,
                hybrid INTEGER,
                top_k INTEGER,
                latency REAL,
                result_count INTEGER,
                payload TEXT,
                created_at TEXT
            );
        """
        )
        conn.commit()
        conn.close()


# 初始化数据库
_init_sqlite()


# --------------------------------------------------------------------
# Query Logger
# --------------------------------------------------------------------
class QueryLogger:
    """
    双写日志：
      1) 写入本地 query.log（JSON Lines）
      2) 写入 SQLite（结构化）
    """

    def __init__(self, log_file: str = LOG_FILE, db_file: str = DB_FILE):
        self.log_file = log_file
        self.db_file = db_file

    # --------------------------------------------------------------
    # 写 JSON 行日志
    # --------------------------------------------------------------
    def log_to_file(self, record: dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            print(f"[QueryLogger] Failed to write log file: {e}")

    # --------------------------------------------------------------
    # 写 SQLite
    # --------------------------------------------------------------
    def log_to_sqlite(self, record: dict[str, Any]) -> None:
        try:
            payload = json.dumps(record, ensure_ascii=False)
            with _sqlite_lock:
                conn = sqlite3.connect(self.db_file)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO query_logs (
                        trace_id,
                        query,
                        hybrid,
                        top_k,
                        latency,
                        result_count,
                        payload,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("trace_id"),
                        record.get("query"),
                        int(record.get("hybrid", False)),
                        record.get("top_k"),
                        record.get("latency_ms"),
                        record.get("result_count"),
                        payload,
                        record.get("timestamp"),
                    ),
                )
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"[QueryLogger] Failed to insert sqlite log: {e}")

    # --------------------------------------------------------------
    # 对外统一接口
    # --------------------------------------------------------------
    def log(self, record: dict[str, Any]) -> None:
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        self.log_to_file(record)
        self.log_to_sqlite(record)


# 单例
query_logger = QueryLogger()
