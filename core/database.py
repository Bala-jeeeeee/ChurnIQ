"""ChurnIQ – SQLite database layer (WAL mode)."""

import sqlite3
import hashlib
import os
import secrets
from config import DB_PATH


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            salt          TEXT    NOT NULL,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS prediction_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            customer_id  TEXT,
            prediction   INTEGER,
            probability  REAL,
            risk_tier    TEXT,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex()


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    try:
        salt = secrets.token_hex(32)
        pw_hash = _hash_password(password, salt)
        conn = _connect()
        conn.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?,?,?,?)",
            (username.strip(), email.strip().lower(), pw_hash, salt),
        )
        conn.commit()
        conn.close()
        return True, "Registration successful."
    except sqlite3.IntegrityError as e:
        msg = str(e)
        if "username" in msg:
            return False, "Username already taken."
        if "email" in msg:
            return False, "Email already registered."
        return False, "Registration failed."
    except Exception as e:
        return False, str(e)


def login_user(identifier: str, password: str) -> tuple[bool, dict | None, str]:
    """identifier can be username OR email."""
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, email, password_hash, salt FROM users "
            "WHERE username=? OR email=?",
            (identifier.strip(), identifier.strip().lower()),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return False, None, "User not found."
        uid, uname, email, pw_hash, salt = row
        if _hash_password(password, salt) != pw_hash:
            return False, None, "Incorrect password."
        return True, {"id": uid, "username": uname, "email": email}, "Login successful."
    except Exception as e:
        return False, None, str(e)


def save_prediction(user_id: int, customer_id: str, prediction: int,
                    probability: float, risk_tier: str) -> None:
    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO prediction_history "
            "(user_id, customer_id, prediction, probability, risk_tier) "
            "VALUES (?,?,?,?,?)",
            (user_id, customer_id, prediction, probability, risk_tier),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_prediction_history(user_id: int) -> list[dict]:
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT customer_id, prediction, probability, risk_tier, created_at "
            "FROM prediction_history WHERE user_id=? ORDER BY created_at DESC LIMIT 100",
            (user_id,),
        )
        rows = cur.fetchall()
        conn.close()
        cols = ["customer_id", "prediction", "probability", "risk_tier", "created_at"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []
