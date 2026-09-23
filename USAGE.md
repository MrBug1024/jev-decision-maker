# JEV Model Gateway

这是一个单进程 JEV 模型服务：网页控制台负责账户与凭证管理，模型能力只通过标准 MCP Streamable HTTP 暴露。

## 架构

- `server.py`：FastAPI 控制台 + MCP 服务，同一端口、同一进程、同一份鉴权。
- `index.html`：登录、注册、Key 管理、模型测试工作台和 MCP 接入说明。
- `jev_gateway.db`：SQLite 数据库，自动创建用户、会话和 Key 表。
- `/mcp`：标准 MCP Streamable HTTP 端点，使用 `Authorization: Bearer <KEY>`。
- `/health`：健康检查，不需要登录。

服务不再提供模型 REST API，也不再启动第二个 MCP 转发进程。

## 启动

### Linux

建议使用 Python 3.10+，生产环境推荐 Python 3.12。

```bash
python3 -m venv .venv
source .venv/bin/activate
cp .env.example .env
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python server.py
```

如果服务器已经安装了与 NVIDIA 驱动匹配的 PyTorch CUDA wheel，`requirements.txt` 会复用它；否则请先按服务器 CUDA 版本从 PyTorch 官方源安装对应 wheel，再执行依赖安装。

两张或多张 GPU 不需要在代码中填写 `0/1`：

```bash
CUDA_VISIBLE_DEVICES=0,1 python server.py
```

程序会根据可见 GPU 数量和实时剩余显存自动选择 CPU、单 GPU 或 `device_map=auto` 多 GPU 分片。大模型建议保留单 worker，避免每个 worker 重复加载模型。

### Windows

```powershell
conda activate open-jev
python -m pip install -r requirements.txt
python server.py
```

首次部署先复制 `.env.example` 为 `.env`，然后按部署机器修改配置。打开 `http://127.0.0.1:8019/`，注册账户后，在“访问 Key”页面创建一个 Key。

局域网访问时，将 `OPEN_JEV_PUBLIC_URL` 设置为其他设备可访问的地址，例如：

```powershell
$env:OPEN_JEV_PUBLIC_URL="http://192.168.1.20:8019"
python server.py
```

## 主要配置

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `OPEN_JEV_MODEL_ID` | 当前 open-jev 模型 | HuggingFace 模型 ID 或本地目录 |
| `OPEN_JEV_MODEL_NAME` | `open-jev-deberta-v3-large` | MCP 返回的模型名 |
| `OPEN_JEV_MODEL_KIND` | `typed_decisions` | 当前模型适配器类型；文本 JEV 使用此值 |
| `OPEN_JEV_MODEL_INPUTS` | `text` | 配置声明的输入能力，逗号分隔；声明不等于适配器已实现 |
| `OPEN_JEV_MODEL_DEVICE` | `auto` | `auto`、`cuda` 或 `cpu` |
| `OPEN_JEV_DEVICE_MAP` | `auto` | 多卡自动分片；也可设为 `none` |
| `OPEN_JEV_MODEL_DTYPE` | `auto` | 自动选择 `bfloat16`、`float16` 或 `float32` |
| `OPEN_JEV_GPU_MEMORY_RESERVE_GIB` | `2` | 每张 GPU 预留显存，防止服务吃满显存 |
| `OPEN_JEV_MODEL_CACHE` | `./.cache/huggingface` | 模型下载缓存目录 |
| `OPEN_JEV_OFFLOAD_DIR` | `./model_offload` | CPU/磁盘 offload 目录 |
| `OPEN_JEV_MAX_CONCURRENT_INFERENCE` | `1` | 并发推理槽位，显存紧张时保持为 1 |
| `OPEN_JEV_MAX_UPLOAD_MB` | `50` | 控制台媒体上传大小上限 |
| `OPEN_JEV_DB_PATH` | `./jev_gateway.db` | SQLite 文件路径 |

`.env` 中的相对路径均相对于项目目录解析。`.env` 不应提交到 Git，生产环境至少修改公开地址、Cookie 安全开关和数据库路径。

## 模型测试工作台

登录控制台后打开“模型测试”，可以使用 `text`、`image`、`audio`、`video` 四种输入模式，填写 Situation、Question 和按行分隔的 Options，并查看结构化结果。该页面调用的是受登录保护的内部控制台接口 `/api/test/decide`，它不是对外提供的模型 REST API；外部 Agent 仍然只能通过 `/mcp` 使用标准 MCP 协议。

当前仓库内置的 `typed_decisions` 适配器只实现文本推理，因此媒体模式会显示上传预览并在提交时返回清晰的“不支持”状态。`akhilaaa/jev-omni` 12B 需要单独的模型 loader、processor 和推理适配器，不能仅通过修改 `OPEN_JEV_MODEL_ID` 就直接启用。适配器完成后，再将配置改为类似：

```dotenv
OPEN_JEV_MODEL_KIND=jev_omni
OPEN_JEV_MODEL_INPUTS=text,image,audio,video
```

Gradio 的 `data`、`fn_index`、`trigger_id`、`session_hash` 是其前端事件协议的内部字段，不作为本项目的 MCP 参数。MCP 继续使用命名对象参数，便于客户端发现和校验：

```json
{
  "situation": "背景文本",
  "questions": [{
    "type": "choice",
    "question": "需要判断的问题",
    "options": ["Yes", "No"]
  }]
}
```

## systemd

仓库提供了可修改路径的模板 `deploy/jev-gateway.service`：

```bash
sudo useradd --system --home /opt/jev-gateway --shell /usr/sbin/nologin jev
sudo cp deploy/jev-gateway.service /etc/systemd/system/jev-gateway.service
sudo systemctl daemon-reload
sudo systemctl enable --now jev-gateway
sudo journalctl -u jev-gateway -f
```

部署前将 service 文件里的用户、项目路径和虚拟环境路径改成实际值。

生产环境建议使用 HTTPS 反向代理，并设置：

```powershell
$env:OPEN_JEV_COOKIE_SECURE="1"
```

## MCP 客户端配置

| 参数 | 值 |
| --- | --- |
| MCP URL | `http://<服务地址>:8019/mcp` |
| 认证 | `Authorization: Bearer <你的 Key>` |
| 工具 | `jev_decide` |

Claude Code 示例：

```bash
claude mcp add --transport http jev-model-gateway http://<服务地址>:8019/mcp \
  --header "Authorization: Bearer <你的 Key>"
```

Python MCP SDK 示例：

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(
    "http://<服务地址>:8019/mcp",
    headers={"Authorization": "Bearer <你的 Key>"},
) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("jev_decide", {
            "situation": "客户的订单到货时损坏，要求退款。",
            "questions": [{
                "type": "choice",
                "question": "接下来应该怎么处理？",
                "options": ["退款", "补充信息", "升级处理"],
            }],
        })
```

`questions` 支持三种类型：`choice` 从选项中分类，`score` 根据有序选项评分，`yes_no` 对一个是非陈述返回为真概率。

模型运行时默认自动选择 CPU、单 GPU 或多 GPU 分片。大模型可通过 `OPEN_JEV_MODEL_ID`、`OPEN_JEV_MODEL_DTYPE` 和 `OPEN_JEV_DEVICE_MAP` 配置；不要使用多个 Uvicorn worker，否则每个 worker 都会加载一份模型。

## 数据与安全

- 密码使用 PBKDF2-HMAC-SHA256 加盐存储。
- API Key 只在创建时显示完整值，数据库只保存 SHA-256 摘要。
- MCP Key 可按账户独立创建、查看前缀、查看最近使用时间和立即吊销。
- 数据库路径默认是项目目录下的 `jev_gateway.db`，可用 `OPEN_JEV_DB_PATH` 修改。
- 默认监听 `0.0.0.0:8019`，对外部署请使用 HTTPS、限制防火墙和反向代理访问范围。

## 调试

如果只需要启动控制台和检查路由，可以跳过模型加载：

```powershell
$env:OPEN_JEV_SKIP_MODEL_LOAD="1"
python server.py
```

这时 `/health` 会显示模型未就绪，MCP 工具调用会明确返回未就绪错误。

# 注意
- typed-decisions==0.0.1 这个依赖不能直接安装，应该使用pip install git+https://github.com/kotoba-lang/typed-decisions方式安装