# gxQwen2api

OpenAI 兼容的千问 OAuth 反代服务，支持多账号池、凭证热重载、内置管理仪表盘。

## 特性

- **多账号池** — 自动扫描凭证目录下所有 `*.json` 凭证文件，注册为独立账号
- **账号启用/禁用** — 在管理面板中手动开关单个账号，无需重启
- **凭证热重载** — 文件 mtime 自动检测 + `SIGHUP` 信号 + 手动重载，改凭证无需重启
- **详细刷新日志** — 记录 refresh 请求 URL、status code、content-type、响应体预览、耗时
- **Round-Robin + 故障转移** — 请求自动轮换健康账号，失败账号进入冷却期
- **轻量管理仪表盘** — 单页 HTML，零 JS 依赖，暗色主题，支持自动刷新
- **内存友好** — 环形日志缓冲 200 条上限，无 opentelemetry 依赖

## 快速启动

### 前置

```bash
# 使用千问 CLI 认证，生成 ~/.qwen/oauth_creds.json
qwen login
```

### Docker

```bash
cp .env.example .env.secret
docker compose --env-file .env.secret up -d
```

### 本地开发

```bash
uv sync
uv run python -m gx_qwen2api.main
```

默认监听 `0.0.0.0:31998`。

## 端点

| 端点 | 说明 |
|---|---|
| `POST /v1/chat/completions` | 聊天补全（流式 + 非流式） |
| `GET /v1/models` | 模型列表 |
| `GET /health` | 健康检查（含账号概览） |
| `GET /healthz` | 最小存活探针 |
| `GET /debug/auth` | 详细账号状态（需认证） |
| `GET /debug/logs` | 近期事件日志（需认证） |
| `GET /admin/` | 管理仪表盘 |

## 管理仪表盘

浏览器打开 `http://localhost:31998/admin/`：

- 所有账号的 token 状态、过期时间、错误数
- **启用 / 禁用** 单个账号
- **强制刷新** 单个账号的 token
- **重载文件** 重新读取磁盘上的凭证
- **扫描目录** 发现新的凭证文件
- **重载全部** 一次性重读所有账号
- 实时事件日志流（彩色编码）
- 自动刷新开关（5s / 15s / 30s / 60s）

如果设置了 `ADMIN_PASSWORD`，首次访问需要登录，密码通过 POST 表单提交（不会出现在 URL 或日志中）。

## 多账号配置

在凭证目录（默认 `~/.qwen`）下放置任意数量的 `*.json` 文件：

```
~/.qwen/
  oauth_creds.json       ← 主账号
  account2.json          ← 第二账号
  service-bot.json       ← 服务账号
```

每个文件格式为标准千问 OAuth 结构：

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "Bearer",
  "resource_url": "https://portal.qwen.ai/v1",
  "expiry_date": 1713000000000
}
```

文件名即为账号 ID，启动时自动扫描注册。运行中可通过管理面板 `Scan Creds Dir` 发现新文件。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PORT` | `31998` | 服务端口 |
| `ADDRESS` | `0.0.0.0` | 绑定地址 |
| `LOG_LEVEL` | `info` | `info` 或 `debug` |
| `LOG_REQUESTS` | `true` | 记录代理请求到 stderr + 仪表盘 |
| `MAX_RETRIES` | `5` | 最大重试次数 |
| `RETRY_DELAY_MS` | `1000` | 基础重试延迟（毫秒） |
| `QWEN_CODE_AUTH_USE` | `true` | 启用千问 OAuth |
| `API_KEY` | _(空)_ | 可选 API 密钥，逗号分隔多个 |
| `DEFAULT_MODEL` | `coder-model` | 默认模型 |
| `CREDS_DIR` | `~/.qwen` | 凭证目录（扫描 `*.json`） |
| `ADMIN_ENABLED` | `true` | 启用管理面板 |
| `ADMIN_PASSWORD` | _(空)_ | 管理员密码（可选，设置后所有管理操作需要认证） |

## 凭证热重载

三种方式无需重启：

1. **SIGHUP** — `kill -HUP <pid>`（Linux/macOS）
2. **API** — 管理面板中的 Reload 按钮
3. **自动** — 每次刷新 token 前检测文件 mtime

## 架构

```
src/gx_qwen2api/
  main.py              — FastAPI 应用，生命周期，SIGHUP 处理
  config.py            — Pydantic 配置
  account_pool.py      — 多账号池：扫描、选择、mtime 重载、启用/禁用
  auth.py              — OAuth 刷新，带详细调试日志
  event_logger.py      — 环形缓冲事件日志（200 条上限）
  message_transform.py  — cache_control + 自定义 system prompt 注入
  models.py            — 模型别名、错误分类
  headers.py           — DashScope 请求头

  routes/
    chat.py            — 代理核心，重试 + 多账号故障转移
    health.py          — /health, /healthz, /debug/*
    models.py          — /v1/models
    admin.py           — 管理 API + 仪表盘页面

  static/
    admin.html         — 单页管理 UI，零依赖
```

## 许可证

MIT
