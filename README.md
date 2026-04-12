# gxQwen2api

OpenAI 兼容的千问 OAuth 反代服务，支持多账号池、凭证热重载、内置管理仪表盘。

## 特性

- **多账号池** — 自动扫描凭证目录下所有 `*.json` 凭证文件，注册为独立账号
- **精准账号状态** — 不再仅依赖过期时间，实时追踪认证错误、刷新失败、冷却等状态
- **Token 校验** — 管理面板提供"校验"按钮，手动验证账号当前是否真实可用
- **账号启用/禁用** — 在管理面板中手动开关单个账号，无需重启
- **凭证热重载** — 文件 mtime 自动检测 + `SIGHUP` 信号 + 手动重载，改凭证无需重启
- **详细刷新日志** — 记录 refresh 请求 URL、status code、content-type、响应体预览、耗时
- **Round-Robin + 故障转移** — 请求自动轮换健康账号，失败账号进入冷却期
- **多语言管理面板** — 默认中文界面，支持一键切换英文，文案集中管理便于扩展
- **轻量管理仪表盘** — 单页 HTML，零 JS 依赖，暗色主题，支持自动刷新、移动端适配
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
- **多语言切换** — 默认中文，右上角一键切换英文（语言偏好保存到 localStorage）
- **启用 / 禁用** 单个账号（禁用时有确认对话框）
- **强制刷新** 单个账号的 token
- **校验** — 手动测试账号当前是否真的可用（调用 API 验证 token），区分权限错误/网络错误/端点异常
- **重载文件** 重新读取磁盘上的凭证
- **扫描目录** 发现新的凭证文件
- **重载全部** 一次性重读所有账号
- 实时事件日志流（彩色编码）
- 自动刷新开关（5s / 15s / 30s / 60s）
- **上传凭证** — 支持拖拽上传，上传后显示成功/失败详情
- **关键时间戳** — 每个账号显示最近校验时间、校验成功时间、校验失败时间
- **冷却倒计时** — 冷却中的账号实时显示剩余秒数
- **登录状态** — 设置了 ADMIN_PASSWORD 后显示登录状态和退出按钮

### 账号状态说明

| 状态 | 说明 |
|---|---|
| `valid` | Token 有效，账号可用 |
| `expiring_soon` | Token 即将过期（< 30 分钟） |
| `expired` | Token 已过期 |
| `auth_error` | 最近请求出现认证错误（真实请求失败后立即标记） |
| `refresh_failed` | Token 刷新失败 |
| `cooldown` | 冷却中（失败后短暂停用） |
| `disabled` | 手动禁用 |
| `no_token` | 无 Token（凭证文件缺少 access_token） |

> **重要**：账号状态不再仅依赖 `expiry_date`。当真实请求出现认证错误时，状态会立即更新为 `auth_error`，管理面板实时显示。如果管理面板显示 valid 但实际已失效，请点击"校验"按钮手动验证。

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

## Token 持久化策略

### 默认行为

当服务刷新 token 后，会**尝试**将更新后的凭证写回磁盘（覆盖原 `*.json` 文件）。
这确保重启后 token 不会丢失。

### Docker 挂载目录权限问题

如果 `CREDS_DIR` 是通过 Docker bind mount 挂载的宿主机目录，容器内的 `nonroot` 用户
可能没有写入权限。此时：

- **服务不会崩溃** — 写入失败时仅记录 warning 日志
- **Token 在内存中正常更新** — 请求不受影响
- **不会修改宿主机文件所有权** — 容器启动时不再递归 `chown` 挂载目录
- **重启后需重新刷新** — 未持久化的 token 在容器重启后失效，需要重新 refresh

**建议**：如果需要持久化，确保挂载目录对容器用户可写：

```bash
# 方法 1：在宿主机上提前设置权限
chmod 777 ~/.qwen

# 方法 2：使用 Docker volume 而非 bind mount
docker volume create qwen_creds
```

### 管理面板上传凭证

通过管理面板上传的凭证文件也会写入 `CREDS_DIR`。同样受上述权限规则约束。
上传后账号会立即在内存中注册，即使写入磁盘失败也不影响当前使用。

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
