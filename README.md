# gx2api

多 Provider OpenAI-compatible API gateway。接入 Qwen OAuth、DeepSeek Web、Freebuff / Codebuff 等渠道，具备多 key 轮询、限流退避、健康状态机和后台管理面板。

> [!IMPORTANT]
> `Qwen OAuth` 免费额度已停用，Qwen 渠道目前仅保留为兼容旧凭证与实验用途，不再建议作为主链路依赖。

## 特性

- **多账号池** — 自动扫描凭证目录下所有 `*.json` 凭证文件，注册为独立账号
- **多渠道账号分组** — 管理面板按 Provider 分组显示，Qwen / Freebuff / DeepSeek 使用不同边框颜色区分
- **精准账号状态** — 不再仅依赖过期时间，实时追踪认证错误、刷新失败、冷却等状态
- **Token 校验** — 管理面板提供"校验"按钮，手动验证账号当前是否真实可用
- **自动预刷新** — 后台定期扫描，在 token 到期前自动续期，无需等待用户请求
- **账号启用/禁用** — 在管理面板中手动开关单个账号，无需重启
- **凭证热重载** — 文件 mtime 自动检测 + `SIGHUP` 信号 + 手动重载，改凭证无需重启
- **详细刷新日志** — 记录 refresh 请求 URL、status code、content-type、响应体预览、耗时
- **Round-Robin + 故障转移** — 请求自动轮换健康账号，失败账号进入冷却期
- **多语言管理面板** — 默认中文界面，支持一键切换英文，文案集中管理便于扩展
- **轻量管理仪表盘** — 单页 HTML，零 JS 依赖，暗色主题，支持自动刷新、移动端适配
- **内存友好** — 环形日志缓冲 200 条上限，无 opentelemetry 依赖
- **模型列表面板** — `/admin/` 可直接查看当前按 Provider 聚合的可用模型列表

## 快速启动

### 新手推荐：Docker 部署

第一次使用，按下面 4 步执行即可：

1. 准备凭证文件：

```bash
mkdir -p data/creds
```

可选来源：

- `Qwen OAuth`：先在宿主机执行 `qwen login`，再把生成的 `oauth_creds.json` 放进 `data/creds/`
- `Freebuff / Codebuff`：把本地 CLI 生成的 `auth-tokens.json` 或单个 `authToken` JSON 放进 `data/creds/`
- `DeepSeek Web`：在 `data/creds/` 下创建 JSON 文件，格式为 `{"email": "...", "password": "..."}`

2. 复制配置文件：

```bash
cp .env.example .env.secret
```

3. 编辑 `.env.secret`，至少确认这几个值：

- `ADMIN_PASSWORD=你的管理面板密码`
- `API_KEY=你的接口密钥`（如果你要对外提供 API，建议设置）
- `HOST_PORT=31998`（默认即可）

4. 用下面这条命令启动：

```bash
docker compose --env-file .env.secret up -d --build
```

启动后访问：

- 管理面板：`http://你的服务器IP:31998/admin/`
- 健康检查：`http://你的服务器IP:31998/healthz`
- 模型接口：`http://你的服务器IP:31998/v1/chat/completions`

> [!IMPORTANT]
> 如果你使用的是 `.env.secret`，启动时必须带上 `--env-file .env.secret`。  
> 只执行 `docker compose up -d` 时，Docker Compose 默认不会读取这个文件。

> [!TIP]
> 默认使用 Docker 命名卷 `qwen_creds` 持久化凭证。这是最省心的方案，不会直接改动宿主机 `~/.qwen` 的权限。

> [!WARNING]
> `Freebuff / Codebuff` 目前按上游实际行为属于**实验性接入**。已知免费模式存在地区限制，如果上游直接返回 `Free mode is not available in your country.`，说明当前出口 IP 所在地区不可用，不是本项目本地凭证解析错误。

> [!IMPORTANT]
> 如果你是从旧版（使用 `~/.qwen` 挂载）升级，第一次启动看到空账号列表是正常现象。请参考[旧版迁移指南](#旧版迁移至-named-volume)把旧凭证迁移进新卷。

### 本地开发

如果你只是本地运行，不用 Docker：

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
- **Provider 分组** — Qwen / Freebuff / DeepSeek 账号分开展示，使用不同颜色边框区分
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
- **当前模型** — 直接查看当前按 Provider 汇总的模型列表
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

在凭证目录（推荐 `./data/creds`）下放置任意数量的 `*.json` 文件：

```
/data/creds/
  oauth_creds.json       ← 主账号
  account2.json          ← 第二账号
  service-bot.json       ← 服务账号
```

### DeepSeek Web 凭证格式

在凭证目录下创建任意 `.json` 文件：

```json
{
  "email": "your-email@example.com",
  "password": "your-password"
}
```

- 启动时会自动识别 `email` + `password` 字段并注册为 DeepSeek 账号
- 首次请求时会自动执行登录流程（含 PoW 挑战）获取 access_token
- 登录成功后 token 会自动写回凭证文件持久化
- 支持管理面板直接添加/删除/登录 DeepSeek 账号

### Qwen OAuth 凭证格式

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "Bearer",
  "resource_url": "https://portal.qwen.ai/v1",
  "expiry_date": 1713000000000
}
```

### Freebuff / Codebuff 凭证格式

支持以下三种格式：

1. 单 token 文件

```json
{
  "authToken": "..."
}
```

2. 带 `default` 包裹的 token

```json
{
  "default": {
    "authToken": "..."
  }
}
```

3. Freebuff CLI 导出的 token 列表

```json
{
  "tokens": [
    {
      "name": "my-account",
      "authToken": "..."
    }
  ]
}
```

第 3 种格式会自动拆成多个账号，例如 `auth-tokens_1`、`auth-tokens_2`。

文件名即为账号 ID（或账号前缀），启动时自动扫描注册。运行中可通过管理面板 `Scan Creds Dir` 发现新文件。

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
| `CREDS_DIR` | `./data/creds` | 凭证目录（扫描 `*.json`） |
| `ADMIN_ENABLED` | `true` | 启用管理面板 |
| `ADMIN_PASSWORD` | _(空)_ | 管理员密码（可选，设置后所有管理操作需要认证） |

## 自动预刷新

> [!NOTE]
> 自动预刷新只对 `Qwen OAuth` 账号生效。`Freebuff / Codebuff` 当前按上游本地 token 方式工作，不走本项目的 refresh_token 流程。

本项目支持后台自动预刷新，在 `access_token` 即将过期前自动续期，无需等待用户请求。

### 工作原理

1. 服务启动后，后台任务定期扫描所有账号
2. 对每个满足条件的账号（enabled + 有 refresh_token + 不在冷却中 + token 剩余时间 < 阈值），自动调用 refresh_token 续期
3. refresh 成功后自动更新 access_token、expiry_date，并持久化到磁盘（如果权限允许）
4. refresh 失败时标记账号状态为 `refresh_failed`，管理面板中可见

### 配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `AUTO_REFRESH_ENABLED` | `true` | 启用自动预刷新 |
| `AUTO_REFRESH_INTERVAL_SECONDS` | `300` | 扫描周期（秒），默认 5 分钟 |
| `AUTO_REFRESH_THRESHOLD_MINUTES` | `30` | 预刷新阈值，token 剩余时间小于此值时触发刷新 |

### 可以自动续期的情况

- 凭证文件中存在有效的 `refresh_token`
- 账号处于 enabled 状态
- 账号不在冷却期

### 必须人工重新登录的情况

- `refresh_token` 本身失效（Qwen 服务端拒绝刷新）
- 凭证文件中没有 `refresh_token`
- 需要更换账号或重新认证

此时管理面板会显示"无 refresh_token，无法自动续期"或"自动刷新失败"，需要用户在本地执行 `qwen login` 重新生成凭证并上传。

### 注意事项

- **本项目不保存账号密码** — 所有认证基于 OAuth refresh_token
- **首次使用**需用户在 Windows / Qwen Code 侧完成 `qwen login` 生成凭证
- 如果 Docker 挂载目录不可写，自动刷新的 token 仅在内存中有效，容器重启后需重新刷新
- 自动预刷新与用户请求的 refresh 逻辑复用同一套代码，不会互相冲突

## 凭证热重载

三种方式无需重启：

1. **SIGHUP** — `kill -HUP <pid>`（Linux/macOS）
2. **API** — 管理面板中的 Reload 按钮
3. **自动** — 每次刷新 token 前检测文件 mtime

## Token 持久化策略

> [!CAUTION]
> **重要安全性警告**：如果凭证无法持久化（写入磁盘失败），系统将进入 **"仅内存 (Memory Only)"** 模式。
> 这意味着在容器重启或服务更新后，所有已生成的 Access Token 和新上传的凭证都会**丢失**，导致服务回退到旧状态甚至不可用。

### 推荐方案：Docker 命名卷 (Named Volume) [最省心]

在 `docker-compose.yml` 中默认启用。Docker 会自动处理文件权限，确保容器内 `nonroot` 用户可以读写。
- **优点**：环境隔离、权限自动处理、升级不丢失（只要卷不删）。
- **缺点**：宿主机无法直接通过文件路径修改凭证。

### 备选方案：管理面板上传 [零配置]

继续使用 `Named Volume`，在宿主机执行 `qwen login` 后，直接打开浏览器访问 `/admin/` 页面，将生成的 `~/.qwen/oauth_creds.json` 拖入上传即可。
- **这是最推荐的“混合使用”方案**，既能利用宿主机的登录工具，又能享受容器内的持久化。

### 进阶方案：宿主机挂载 (Bind Mount) [有风险]

如果你一定要直接挂载宿主机目录：

```yaml
# docker-compose.yml
services:
  gx2api:
    volumes:
      - ~/.qwen:/app/data/creds
```

> [!WARNING]
> **权限警告**：由于容器运行在 `nonroot` (UID 999) 用户下，如果宿主机目录不可写，Token 将无法持久化。
> 1. **不推荐** 使用 `chown -R 999:999 ~/.qwen`，因为这会破坏宿主机工具（如 `qwen login`）的权限。
> 2. **注意**：如果你将目录挂载为只读（`:ro`），则管理面板的所有更新（刷新 Token、上传凭证）都**无法写回磁盘**，仅在内存中生效。

## 旧版迁移至 Named Volume

如果你之前使用的是 `~/.qwen` 挂载，只需执行以下指令将数据一键搬入新卷：

```bash
# 1. 确保容器已启动
docker compose up -d

# 2. 将宿主机 ~/.qwen 下的所有 JSON 拷贝进容器内的卷路径
# 这会自动处理权限，且通过 Compose 明确指定服务，不会误伤其他项目
docker compose cp ~/.qwen/. gx-qwen-api:/app/data/creds

# 3. 在管理面板点击 "Scan Creds Dir" 即可看到新账号
```

### 监控持久化状态

- **管理面板**：如果写入失败，账号卡片上会显示黄色感叹号 `⚠️ 仅内存`。
- **健康检查**：`/health` 接口会包含 `persistence_warnings` 计数。
- **日志**：Stderr 会记录 `Cannot write ... — permission denied` 的详细错误。

## 架构

```
src/gx_qwen2api/
  main.py              — FastAPI 应用，生命周期，SIGHUP 处理
  config.py            — Pydantic 配置
  account_pool.py      — 多账号池：扫描、选择、mtime 重载、启用/禁用
  auth.py              — OAuth 刷新，带详细调试日志
  providers/           — Provider 适配层（Qwen / Freebuff / DeepSeek）
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
