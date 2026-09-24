"""JEV Model Gateway.

One process exposes the web console and the standard MCP Streamable HTTP
endpoint. User accounts, sessions, and API keys are stored in SQLite; the
model is loaded once for the whole process.

Start with:
    conda activate open-jev
    python server.py

Then open http://127.0.0.1:8019/ and create an account. The MCP endpoint is
available at http://127.0.0.1:8019/mcp.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from pathlib import Path
import secrets
import sqlite3
import time
import traceback
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

from config import settings
from model_runtime import ModelRuntime

SESSION_COOKIE = "jev_session"
logger = logging.getLogger("jev.gateway")
runtime = ModelRuntime(settings)
MEDIA_MIME_PREFIXES = {
    "image": "image/",
    "audio": "audio/",
    "video": "video/",
}
MEDIA_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"},
    "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"},
}


def now() -> int:
    return int(time.time())


def db_connection() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(settings.database_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_database() -> None:
    with db_connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_login_at INTEGER
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_hash TEXT NOT NULL UNIQUE,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL,
                last_used_at INTEGER,
                revoked_at INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_token_hash
                ON sessions(token_hash);
            CREATE INDEX IF NOT EXISTS idx_api_keys_user_id
                ON api_keys(user_id);
            CREATE INDEX IF NOT EXISTS idx_api_keys_active
                ON api_keys(key_hash, revoked_at);
            """
        )
        db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now(),))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, settings.password_iterations
    )
    return "pbkdf2_sha256${}${}${}".format(
        settings.password_iterations,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(derived).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            base64.urlsafe_b64decode(salt_text.encode("ascii")),
            int(iterations),
        )
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        return hmac.compare_digest(derived, expected)
    except (ValueError, TypeError):
        return False


def user_payload(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def create_session(user_id: int) -> str:
    raw_token = secrets.token_urlsafe(32)
    with db_connection() as db:
        db.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (digest(raw_token), user_id, now(), now() + settings.session_ttl),
        )
    return raw_token


def find_user_by_session(raw_token: str | None) -> sqlite3.Row | None:
    if not raw_token:
        return None
    with db_connection() as db:
        return db.execute(
            """
            SELECT users.id, users.username, users.created_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ? AND sessions.expires_at > ?
            """,
            (digest(raw_token), now()),
        ).fetchone()


def get_active_key(raw_key: str) -> sqlite3.Row | None:
    if not raw_key:
        return None
    with db_connection() as db:
        row = db.execute(
            """
            SELECT api_keys.id, api_keys.user_id, api_keys.name,
                   users.username, api_keys.created_at
            FROM api_keys
            JOIN users ON users.id = api_keys.user_id
            WHERE api_keys.key_hash = ? AND api_keys.revoked_at IS NULL
            """,
            (digest(raw_key),),
        ).fetchone()
        if row:
            db.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now(), row["id"]))
        return row


class MCPKeyVerifier(TokenVerifier):
    """Validate JEV API keys through the same SQLite store as the console."""

    async def verify_token(self, token: str) -> AccessToken | None:
        key = get_active_key(token)
        if not key:
            return None
        return AccessToken(
            token=token,
            client_id=f"jev-key-{key['id']}",
            subject=str(key["user_id"]),
            scopes=["jev:decide"],
            resource=settings.mcp_resource_url,
            claims={"username": key["username"], "key_name": key["name"]},
        )


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=128)


class CreateKeyRequest(BaseModel):
    name: str = Field(default="我的 MCP Key", min_length=1, max_length=60)


class DecisionQuestion(BaseModel):
    """Public MCP question schema; the runtime maps it to JEV's wire format."""

    type: Literal["choice", "score", "yes_no"] = Field(
        description="choice, score, or yes_no"
    )
    question: str = Field(min_length=1, max_length=4_000)
    options: list[str] | None = Field(
        default=None,
        description="Required for choice and score; score options are ordered low-to-high.",
    )

    @model_validator(mode="after")
    def validate_options(self) -> "DecisionQuestion":
        options = self.options or []
        if self.type == "choice" and not 2 <= len(options) <= 255:
            raise ValueError("choice requires 2 to 255 options")
        if self.type == "score" and not 2 <= len(options) <= 10:
            raise ValueError("score requires 2 to 10 ordered options")
        if self.type in {"choice", "score"}:
            if any(not item.strip() for item in options):
                raise ValueError("options must be non-empty strings")
            if len(set(options)) != len(options):
                raise ValueError("options must be unique")
        return self

    def to_model_question(self) -> dict[str, Any]:
        question = {
            "type": "noul" if self.type == "yes_no" else self.type,
            "instructions": self.question,
        }
        if self.type in {"choice", "score"}:
            question["options"] = self.options
        return question


def set_session_cookie(response: JSONResponse, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


def require_user(jev_session: str | None = Cookie(default=None)) -> sqlite3.Row:
    user = find_user_by_session(jev_session)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def model_status() -> dict[str, Any]:
    return {
        "model": settings.model_name,
        "model_id": settings.model_id,
        "max_upload_mb": settings.max_upload_mb,
        **runtime.status(),
    }


def parse_options(value: str) -> list[str]:
    return [item.strip() for item in value.splitlines() if item.strip()]


async def validate_upload(upload: UploadFile | None, input_type: str) -> tuple[bytes, str]:
    if upload is None or not upload.filename:
        raise HTTPException(status_code=400, detail=f"{input_type} 输入需要上传一个文件")

    content_type = (upload.content_type or "").lower()
    suffix = Path(upload.filename).suffix.lower()
    expected_prefix = MEDIA_MIME_PREFIXES[input_type]
    if not content_type.startswith(expected_prefix) and suffix not in MEDIA_EXTENSIONS[input_type]:
        raise HTTPException(
            status_code=415,
            detail=f"文件类型与 {input_type} 不匹配，请上传 {expected_prefix[:-1]} 文件",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过 {settings.max_upload_mb} MB 上传限制",
        )
    return content, content_type or "application/octet-stream"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    if not settings.skip_model_load:
        try:
            print(f"Loading {settings.model_id} with automatic device placement ...")
            runtime.load()
            print(f"Model loaded. Runtime: {runtime.status()}")
        except Exception as exc:  # The console can still report a useful startup error.
            runtime.set_error(str(exc))
            traceback.print_exc()
            print(f"Model failed to load: {exc}")
    else:
        runtime.set_error("Model loading skipped by OPEN_JEV_SKIP_MODEL_LOAD")

    async with mcp_server.session_manager.run():
        yield

    runtime.model = None


def build_mcp_server() -> FastMCP:
    auth = AuthSettings(
        issuer_url=AnyHttpUrl(settings.public_base_url),
        resource_server_url=AnyHttpUrl(settings.mcp_resource_url),
        required_scopes=["jev:decide"],
        validate_token_resource=False,
    )
    server = FastMCP(
        name="jev-model-gateway",
        instructions=(
            f"Structured decisions from the {settings.model_name} model. Use jev_decide "
            "for structured classification, ordered scoring, and yes/no judgments. "
            "The model returns probabilities instead of generated text."
        ),
        token_verifier=MCPKeyVerifier(),
        auth=auth,
        host=settings.host,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )

    @server.tool()
    def jev_decide(
        situation: Annotated[str, Field(min_length=1, max_length=16_000)],
        questions: Annotated[list[DecisionQuestion], Field(min_length=1, max_length=64)],
    ) -> dict[str, Any]:
        """Evaluate a situation with one or more structured questions.

        Use choice for categorical classification, score for ordered levels,
        and yes_no for a binary statement. Results preserve input order.
        """
        if get_access_token() is None:
            raise RuntimeError("MCP authentication is required")
        try:
            normalized = [question.to_model_question() for question in questions]
            return {"model": settings.model_name, "results": runtime.decide(situation, normalized)}
        except Exception as exc:
            raise RuntimeError(f"JEV inference failed: {exc}") from exc

    return server


mcp_server = build_mcp_server()
mcp_http_app = mcp_server.streamable_http_app()

app = FastAPI(title="JEV Model Gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTMLResponse(
        settings.index_html.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "jev-model-gateway", **model_status()}


@app.post("/api/auth/register")
def register(request: RegisterRequest) -> JSONResponse:
    username = request.username.strip()
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少需要 2 个字符")
    try:
        with db_connection() as db:
            cursor = db.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(request.password), now()),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="用户名已存在")

    response = JSONResponse({"user": {"id": user_id, "username": username}})
    set_session_cookie(response, create_session(int(user_id)))
    return response


@app.post("/api/auth/login")
def login(request: LoginRequest) -> JSONResponse:
    with db_connection() as db:
        user = db.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ? COLLATE NOCASE",
            (request.username.strip(),),
        ).fetchone()
        if not user or not verify_password(request.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="用户名或密码不正确")
        db.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now(), user["id"]))

    response = JSONResponse({"user": user_payload(user)})
    set_session_cookie(response, create_session(user["id"]))
    return response


@app.post("/api/auth/logout")
def logout(jev_session: str | None = Cookie(default=None)) -> JSONResponse:
    if jev_session:
        with db_connection() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (digest(jev_session),))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/auth/me")
def me(user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:
    return {"user": user_payload(user)}


@app.get("/api/status")
def status(user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:
    with db_connection() as db:
        count = db.execute(
            "SELECT COUNT(*) AS count FROM api_keys WHERE user_id = ? AND revoked_at IS NULL",
            (user["id"],),
        ).fetchone()["count"]
    return {**model_status(), "mcp_endpoint": settings.mcp_resource_url, "active_key_count": count}


@app.post("/api/test/decide")
async def test_decide(
    input_type: Literal["text", "image", "audio", "video"] = Form("text"),
    situation: str = Form(..., min_length=1, max_length=16_000),
    question: str = Form(..., min_length=1, max_length=4_000),
    options: str = Form(""),
    question_type: Literal["choice", "score", "yes_no"] = Form("choice"),
    media: UploadFile | None = File(default=None),
    user: sqlite3.Row = Depends(require_user),
) -> dict[str, Any]:
    """Run a browser-only test request; external model access remains MCP-only."""
    del user
    if not situation.strip():
        raise HTTPException(status_code=422, detail="Situation 不能为空。")
    if not question.strip():
        raise HTTPException(status_code=422, detail="Question 不能为空。")
    current_status = runtime.status()
    if not current_status.get("ready"):
        raise HTTPException(
            status_code=503,
            detail=current_status.get("error") or "模型尚未就绪，请稍后重试",
        )
    if not runtime.supports_input_type(input_type):
        supported = ", ".join(current_status.get("supported_inputs") or []) or "无"
        raise HTTPException(
            status_code=422,
            detail=f"当前模型暂不支持 {input_type} 输入，已实现能力：{supported}",
        )

    media_bytes: bytes | None = None
    content_type: str | None = None
    filename: str | None = None
    if input_type != "text":
        media_bytes, content_type = await validate_upload(media, input_type)
        filename = media.filename if media else None

    option_values = parse_options(options)
    try:
        decision = DecisionQuestion(
            type=question_type,
            question=question.strip(),
            options=None if question_type == "yes_no" else option_values,
        )
    except ValidationError as exc:
        message = exc.errors(include_url=False)[0]["msg"]
        if "unique" in message:
            detail = "选项不能重复，请检查每一行。"
        elif question_type == "score":
            detail = "Score 需要 2 到 10 个选项，并按从低到高排列。"
        elif question_type == "choice":
            detail = "Choice 至少需要 2 个选项，每行填写一个。"
        else:
            detail = message.removeprefix("Value error, ")
        raise HTTPException(status_code=422, detail=detail) from exc

    try:
        results = await run_in_threadpool(
            runtime.test_decide,
            input_type,
            situation.strip(),
            [decision.to_model_question()],
            media_bytes,
            filename,
            content_type,
        )
    except Exception as exc:
        error_id = secrets.token_hex(4)
        logger.exception("JEV inference failed (request_id=%s)", error_id)
        error_text = str(exc).lower()
        if "out of memory" in error_text or "cuda error: memory" in error_text:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"模型显存不足（请求 {error_id}）。请降低并发或上下文长度、"
                    "释放其他 GPU 占用，或调整模型量化/设备放置配置。"
                ),
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=f"模型推理失败（请求 {error_id}，{type(exc).__name__}）。请查看服务端日志。",
        ) from exc
    return {
        "model": settings.model_name,
        "model_kind": settings.model_kind,
        "input_type": input_type,
        "filename": filename,
        "content_type": content_type,
        "results": results,
    }


@app.get("/api/keys")
def list_keys(user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:
    with db_connection() as db:
        rows = db.execute(
            """
            SELECT id, name, key_prefix, created_at, last_used_at
            FROM api_keys
            WHERE user_id = ? AND revoked_at IS NULL
            ORDER BY created_at DESC
            """,
            (user["id"],),
        ).fetchall()
    return {
        "keys": [
            {
                "id": row["id"],
                "name": row["name"],
                "prefix": row["key_prefix"] + "...",
                "created_at": row["created_at"],
                "last_used_at": row["last_used_at"],
            }
            for row in rows
        ]
    }


@app.post("/api/keys")
def create_key(request: CreateKeyRequest, user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:
    raw_key = "jev_live_" + secrets.token_urlsafe(32)
    with db_connection() as db:
        cursor = db.execute(
            """
            INSERT INTO api_keys (user_id, name, key_prefix, key_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user["id"], request.name.strip(), raw_key[:18], digest(raw_key), now()),
        )
    return {
        "id": cursor.lastrowid,
        "name": request.name.strip(),
        "key": raw_key,
        "warning": "完整 Key 只显示这一次，请立即保存。",
    }


@app.delete("/api/keys/{key_id}")
def revoke_key(key_id: int, user: sqlite3.Row = Depends(require_user)) -> dict[str, Any]:
    with db_connection() as db:
        result = db.execute(
            """
            UPDATE api_keys SET revoked_at = ?
            WHERE id = ? AND user_id = ? AND revoked_at IS NULL
            """,
            (now(), key_id, user["id"]),
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Key 不存在或已经吊销")
    return {"ok": True}


# Keep the MCP app after the console routes so the root UI and /api endpoints
# are handled by FastAPI while the protocol endpoint remains owned by the SDK.
# Mounting at / preserves the exact /mcp endpoint instead of redirecting it.
app.mount("/", mcp_http_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
