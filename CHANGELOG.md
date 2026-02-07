# Changelog

## [Unreleased]

### Fixed
- **[P0] API Key 日志泄露修复** (`src/main.rs`)
  - info 级别不再打印 API Key 前半段，仅显示末 4 位和长度
  - 完整前缀仅在 `sensitive-logs` feature 的 debug 级别输出
- **[P2] 占位工具大小写变体重复插入** (`src/anthropic/converter.rs`)
  - `collect_history_tool_names` 改为小写去重，避免 `read`/`Read` 等变体重复
  - 占位工具 push 后同步更新 `existing_tool_names` 集合
- **[P2] 最后一条消息角色校验** (`src/anthropic/converter.rs`, `src/anthropic/handlers.rs`)
  - `convert_request` 显式校验最后一条消息 role 必须为 `user`
  - 新增 `InvalidLastMessageRole` 错误变体，映射为 400 响应
- **[P2] 凭据回写原子性** (`src/kiro/token_manager.rs`)
  - `persist_credentials` 改为临时文件 + `rename` 原子替换
  - 新增 `resolve_symlink_target` 辅助函数：优先 `canonicalize`，失败时用 `read_link` 解析 symlink
  - 保留原文件权限，防止 umask 导致凭据文件权限放宽
  - Windows 兼容：`rename` 前先删除已存在的目标文件
  - 避免进程崩溃或并发调用导致凭据文件损坏

### Changed
- 重构 README.md 配置文档，提升新用户上手体验
  - 明确配置文件默认路径：当前工作目录（或通过 `-c`/`--config` 和 `--credentials` 参数指定）
  - 添加 JSON 注释警告：移除所有带 `//` 注释的示例，提供可直接复制的配置
  - 修正字段必填性：仅 `apiKey` 为必填，其他字段均有默认值
  - 新增命令行参数说明表格（`-c`, `--credentials`, `-h`, `-V`）
  - 补充遗漏的 `credentialRpm` 字段说明（凭据级 RPM 限流）
  - 使用表格形式展示配置字段，标注必填/可选和默认值
- 优化 debug 日志中请求体的输出长度 (`src/anthropic/handlers.rs`)
  - 新增 `truncate_middle()` 函数：截断字符串中间部分，保留头尾各 1200 字符
  - 正确处理 UTF-8 多字节字符边界，不会截断中文
  - 仅在启用 `sensitive-logs` feature 时生效，减少日志噪音

### Fixed
- 限制 `max_tokens` 最大值为 32000（Kiro 上游限制）
  - 当用户设置超出限制的值时自动调整为 32000
  - 记录 WARN 级别日志，包含原始值和调整后的值
  - 涉及文件：`src/anthropic/handlers.rs`
- 工具输入 JSON 解析失败时的日志输出改为受 `sensitive-logs` feature 控制
  - 默认仅输出 `buffer_len` 和 `request_body_bytes`（长度信息）
  - 启用 `--features sensitive-logs` 时输出完整 `buffer` 和 `request_body`
  - 涉及文件：`src/anthropic/handlers.rs`
- 修复 Kiro 上游请求兼容性问题
  - 空 content（仅 tool_result/image）时使用占位符避免 400
  - 规范化工具 JSON Schema、补全空 description
  - 禁用 reqwest 系统代理探测（仅支持显式 `config.proxy_url`）
  - 新增离线诊断脚本：`tools/diagnose_improper_request.py`
  - 涉及文件：`src/anthropic/converter.rs`、`src/http_client.rs`、`tools/diagnose_improper_request.py`、`.gitignore`

### Changed
- 优化 400 Bad Request "输入过长" 错误的日志输出 (`src/kiro/provider.rs`)
  - 对于 `CONTENT_LENGTH_EXCEEDS_THRESHOLD` / `Input is too long` 错误，不再输出完整请求体（太占空间且无调试价值）
  - 改为记录 `request_body_bytes`（字节数）和 `estimated_input_tokens`（估算 token 数）
  - 新增 `estimate_tokens()` 函数：基于 CJK/非 CJK 字符比例估算 token 数量
    - CJK 字符（中/日/韩）: token 数 = 字符数 / 1.5
    - 其他字符（英文等）: token 数 = 字符数 / 3.5
  - 新增 `is_input_too_long()` 和 `is_cjk_char()` 辅助函数

### Added
- 新增多维度设备指纹系统 (`src/kiro/fingerprint.rs`)
  - 每个凭据生成独立的确定性指纹，模拟真实 Kiro IDE 客户端
  - 支持 10+ 维度设备信息：SDK 版本、Kiro 版本、Node.js 版本、操作系统、屏幕分辨率、CPU 核心数、时区等
  - 提供 `user_agent()` 和 `x_amz_user_agent()` 方法构建请求头
  - 参考 CLIProxyAPIPlus 实现，降低被检测风险

- 新增精细化速率限制系统 (`src/kiro/rate_limiter.rs`)
  - 每日请求限制（默认 500 次/天）
  - 请求间隔控制（1-2 秒 + 30% 抖动）
  - 指数退避策略（30s → 5min，倍数 1.5）
  - 暂停检测（关键词匹配：suspended, banned, quota exceeded 等）

- 新增独立冷却管理模块 (`src/kiro/cooldown.rs`)
  - 分类冷却原因（7 种类型：速率限制、账户暂停、配额耗尽、Token 刷新失败等）
  - 差异化冷却时长：短冷却（1-5 分钟）vs 长冷却（1-24 小时）
  - 递增冷却机制（连续触发时延长冷却时间）
  - 自动清理过期冷却

- 新增后台 Token 刷新模块 (`src/kiro/background_refresh.rs`)
  - 独立后台任务定期检查即将过期的 Token
  - 支持批量并发刷新（信号量控制）
  - 可配置检查间隔、批处理大小、并发数
  - 优雅关闭机制

- `MultiTokenManager` 新增增强方法
  - `get_fingerprint()`: 获取凭据的设备指纹
  - `is_credential_available()`: 综合检查凭据可用性（未禁用、未冷却、未超速率限制）
  - `set_credential_cooldown()` / `clear_credential_cooldown()`: 冷却管理
  - `get_expiring_credential_ids()`: 获取即将过期的凭据列表
  - `start_background_refresh()`: 启动后台 Token 刷新任务
  - `refresh_token_for_credential()`: 带优雅降级的 Token 刷新
  - `record_api_success()` / `record_api_failure()`: 更新速率限制器状态

### Changed
- `CredentialEntry` 结构体新增 `fingerprint` 字段，每个凭据独立生成设备指纹

### Fixed
- 修复 IDC 凭据 `fetch_profile_arn` 在某些 region 返回 `UnknownOperationException` 的问题
  - 新增 `ListAvailableCustomizations` API 作为 `ListProfiles` 的回退方案
  - 支持多 region 尝试：先尝试用户配置的 region，失败则回退到 `us-east-1`
  - 涉及文件：`src/kiro/token_manager.rs`

- 修复 `start_background_refresh` 后台刷新器生命周期问题（Codex Review P1）
  - 问题：`refresher` 作为局部变量在函数返回后被 drop，导致后台任务立即停止
  - 解决：方法现在返回 `Arc<BackgroundRefresher>`，调用方需保持引用以维持任务运行
  - 涉及文件：`src/kiro/token_manager.rs`

- 修复 `calculate_backoff` 退避时间可能超过配置上限的问题（Codex Review P2）
  - 问题：添加抖动后未再次进行上限约束，可能导致实际等待时间超过 `backoff_max_ms`
  - 解决：在添加抖动后再进行 `.min(max)` 约束
  - 涉及文件：`src/kiro/rate_limiter.rs`

- 改进 `persist_credentials` 并发写入安全性（Codex Review P1）
  - 问题：在锁外执行文件写入可能导致并发写入时旧快照覆盖新数据
  - 解决：在持有 entries 锁的情况下完成序列化，确保快照一致性
  - 涉及文件：`src/kiro/token_manager.rs`
- 修复 IDC 凭据返回 403 "The bearer token included in the request is invalid" 的问题
  - 根本原因：`profile_arn` 只从第一个凭据获取并存储在全局 `AppState` 中，当使用 IDC 凭据时，Bearer Token 来自 IDC 凭据，但 `profile_arn` 来自第一个凭据（可能是 Social 类型），导致 Token 和 profile_arn 不匹配
  - 解决方案 1：在 `call_api_with_retry` 中动态注入当前凭据的 `profile_arn`，确保 Token 和 profile_arn 始终匹配
  - 解决方案 2：IDC Token 刷新后自动调用 `ListProfiles` API 获取 `profileArn`（IDC 的 OIDC 刷新不返回此字段）
  - 新增 `inject_profile_arn()` 辅助方法，解析请求体 JSON 并覆盖 `profileArn` 字段
  - 新增 `fetch_profile_arn()` 方法，通过 CodeWhisperer ListProfiles API 获取 profileArn
  - 涉及文件：`src/kiro/provider.rs`, `src/kiro/token_manager.rs`

### Added
- 新增批量导入 token.json 功能
  - 后端：新增 `POST /api/admin/credentials/import-token-json` 端点
  - 支持解析官方 token.json 格式（含 `provider`、`refreshToken`、`clientId`、`clientSecret` 等字段）
  - 按 `provider` 字段自动映射 `authMethod`（BuilderId → idc, IdC → idc, Social → social）
  - 支持 dry-run 预览模式，返回详细的导入结果（成功/跳过/无效）
  - 通过 refreshToken 前缀匹配自动去重，避免重复导入
  - 前端：新增"导入 token.json"对话框组件
  - 支持拖放上传 JSON 文件或直接粘贴 JSON 内容
  - 三步流程：输入 → 预览 → 结果
  - 涉及文件：
    - `src/admin/types.rs`（新增 `TokenJsonItem`、`ImportTokenJsonRequest`、`ImportTokenJsonResponse` 等类型）
    - `src/admin/service.rs`（新增 `import_token_json()` 方法）
    - `src/admin/handlers.rs`（新增 `import_token_json` handler）
    - `src/admin/router.rs`（添加路由）
    - `src/kiro/token_manager.rs`（新增 `has_refresh_token_prefix()` 方法）
    - `admin-ui/src/types/api.ts`（新增导入相关类型）
    - `admin-ui/src/api/credentials.ts`（新增 `importTokenJson()` 函数）
    - `admin-ui/src/hooks/use-credentials.ts`（新增 `useImportTokenJson()` hook）
    - `admin-ui/src/components/import-token-json-dialog.tsx`（新建）
    - `admin-ui/src/components/dashboard.tsx`（添加导入按钮）

### Fixed
- 修复字符串切片在多字节字符中间切割导致 panic 的风险（DoS 漏洞）
  - `generate_fingerprint()` 和 `has_refresh_token_prefix()` 使用 `floor_char_boundary()` 安全截断
  - 涉及文件：`src/admin/service.rs`, `src/kiro/token_manager.rs`
- 修复日志截断在多字节字符中间切割导致 panic 的问题
  - `truncate_for_log()` 使用 `floor_char_boundary()` 安全截断 UTF-8 字符串
  - 删除 `stream.rs` 中冗余的 `find_char_boundary()` 函数，直接使用标准库方法
  - 涉及文件：`src/kiro/provider.rs`, `src/anthropic/stream.rs`
- 移除历史消息中孤立的 tool_use（无对应 tool_result）
  - Kiro API 要求 tool_use 必须有配对的 tool_result，否则返回 400 Bad Request
  - 新增 `remove_orphaned_tool_uses()` 函数清理孤立的 tool_use
  - 涉及文件：`src/anthropic/converter.rs`
- 修复 `/cc/v1/messages` 缓冲流 ping 定时器首次立即触发的问题
  - 将 `interval()` 改为 `interval_at(Instant::now() + ping_period, ping_period)`
  - 现在首个 ping 会在 25 秒后触发，与 `/v1/messages` 行为一致
  - 涉及文件：`src/anthropic/handlers.rs`
- 修复 Clippy `collapsible_if` 警告
  - 使用 let-chains 语法合并嵌套 if 语句
  - 涉及文件：`src/anthropic/stream.rs`

### Changed
- 增强 400 Bad Request 错误日志，记录完整请求信息
  - 移除请求体截断限制，记录完整的 `request_body`
  - 新增 `request_url` 和 `request_headers` 字段
  - 新增 `format_headers_for_log()` 辅助函数，对 Authorization 头进行脱敏处理
  - 删除不再使用的 `truncate_for_log()` 函数（YAGNI 原则）
  - 涉及文件：`src/kiro/provider.rs`
- 改进凭据选择算法：同优先级内实现负载均衡
  - 第一优先级：使用次数最少
  - 第二优先级：余额最多（使用次数相同时）
  - 第三优先级：轮询选择（使用次数和余额完全相同时，避免总选第一个）
  - 新增 `selection_rr` 原子计数器用于轮询抖动
  - 新增 `select_best_candidate_id()` 方法实现三级排序逻辑
  - 涉及文件：`src/kiro/token_manager.rs`

### Fixed
- 修复测试代码使用 `serde_json::json!` 构造 Tool 对象导致的类型不匹配问题
  - 改用 `Tool` 结构体直接构造，确保类型安全
  - 涉及文件：`src/anthropic/websearch.rs`
- 修复 `select_best_candidate_id()` 中 NaN 余额处理问题
  - 在评分阶段将 NaN/Infinity 余额归一化为 0.0
  - 避免 NaN 被 `total_cmp` 视为最大值导致错误的凭据选择
  - 避免 NaN 导致 `scored` 被完全过滤后除零 panic
  - 涉及文件：`src/kiro/token_manager.rs`

### Added
- 新增 `system` 字段格式兼容性支持（`src/anthropic/types.rs`）
  - 支持字符串格式：`"system": "You are a helpful assistant"`（new-api 等网关添加的系统提示词）
  - 支持数组格式：`"system": [{"type": "text", "text": "..."}]`（Claude Code 原生格式）
  - 自动将字符串格式转换为单元素数组，保持内部处理一致性
  - 新增 6 个单元测试验证格式兼容性
- 新增请求体大小限制：50MB（`DefaultBodyLimit::max(50 * 1024 * 1024)`）
  - 涉及文件：`src/anthropic/router.rs`

### Changed
- 调整全局禁用恢复时间：`GLOBAL_DISABLE_RECOVERY_MINUTES` 从 10 分钟降至 5 分钟
  - 加快模型暂时不可用后的自动恢复速度
- 调整总重试次数硬上限：`MAX_TOTAL_RETRIES` 从 5 降至 3
  - 进一步减少无效重试开销，加快故障转移速度
- 余额初始化改为顺序查询，每次间隔 0.5 秒避免触发限流
  - 从并发查询改为顺序查询（`initialize_balances()`）
  - 移除 30 秒整体超时机制
  - 涉及文件：`src/kiro/token_manager.rs`

### Fixed
- 修复 assistant 消息仅包含 tool_use 时 content 为空导致 Kiro API 报错的问题
  - 当 text_content 为空且存在 tool_uses 时，使用 "OK" 作为占位符
  - 涉及文件：`src/anthropic/converter.rs`
- 修复 `MODEL_TEMPORARILY_UNAVAILABLE` 错误检测逻辑未实际调用的问题
  - 在 `call_mcp()` 和 `call_api()` 中添加错误检测和熔断触发逻辑
  - 移除 `report_model_unavailable()` 和 `disable_all_credentials()` 的 `#[allow(dead_code)]` 标记
  - 现在当检测到该错误时会正确触发全局熔断机制

### Added
- 新增 WebSearch 工具支持（`src/anthropic/websearch.rs`）
  - 实现 Anthropic WebSearch 请求到 Kiro MCP 的转换
  - 支持 SSE 流式响应，生成完整的搜索结果事件序列
  - 自动检测纯 WebSearch 请求（tools 仅包含 web_search）并路由到专用处理器
- 新增 MCP API 调用支持（`src/kiro/provider.rs`）
  - 新增 `call_mcp()` 方法，支持 WebSearch 等工具调用
  - 新增 `mcp_url()` 和 `build_mcp_headers()` 方法
  - 完整的重试和故障转移逻辑
- 新增凭据级 `region` 字段，用于 OIDC token 刷新时指定 endpoint 区域
  - 未配置时回退到 config.json 的全局 region
  - API 调用仍使用 config.json 的 region
- 新增凭据级 `machineId` 字段，支持每个凭据使用独立的机器码
  - 支持 64 字符十六进制和 UUID 格式（自动标准化）
  - 未配置时回退到 config.json 的 machineId，都未配置时由 refreshToken 派生
  - 启动时自动补全并持久化到配置文件
- 新增 GitHub Actions Docker 构建工作流（`.github/workflows/docker-build.yaml`）
  - 支持 linux/amd64 和 linux/arm64 双架构
  - 推送到 GitHub Container Registry

### Changed
- 版本号升级至 2026.1.5
- TLS 库从 native-tls 切换至 rustls（reqwest 依赖调整）
- `authMethod` 自动推断：未指定时根据是否有 clientId/clientSecret 自动判断为 idc 或 social
- 移除 web_search/websearch 工具过滤（`is_unsupported_tool` 现在返回 false）

### Fixed
- 修复 machineId 格式兼容性问题，支持 UUID 格式自动转换为 64 字符十六进制

### Removed
- 移除 `current_id` 概念（后端和前端）
  - 后端：移除 `MultiTokenManager.current_id` 字段和相关方法（`switch_to_next`、`select_highest_priority`、`select_by_balance`、`credentials`）
  - 后端：移除 `ManagerSnapshot.current_id` 字段
  - 后端：移除 `CredentialStatusItem.is_current` 字段
  - 前端：移除 `CredentialsStatusResponse.currentId` 和 `CredentialStatusItem.isCurrent`
  - 原因：多用户并发访问时，"当前凭据"概念无意义，凭据选择由 `acquire_context_for_user()` 动态决定

### Added
- 新增启动时余额初始化功能
  - `initialize_balances()`: 启动时并发查询所有凭据余额并更新缓存
  - 整体超时 30 秒，避免阻塞启动流程
  - 初始化失败或超时时输出警告日志

### Changed
- 改进凭据选择算法：从单一"使用次数最少"改为两级排序
  - 第一优先级：使用次数最少
  - 第二优先级：余额最多（使用次数相同时）
  - 未初始化余额的凭据会被降级处理，避免被优先选中
- 移除前端"当前活跃"凭据展示
  - 前端：移除凭据卡片的"当前"高亮和 Badge
  - 前端：移除 Dashboard 中的"当前活跃"统计卡片
  - 统计卡片布局从 3 列调整为 2 列

### Added
- 新增 `sensitive-logs` feature flag，显式启用才允许打印潜在敏感信息（仅用于排障）
  - 默认关闭：Kiro 请求体只输出长度，凭证只输出摘要信息
  - 启用方式：`cargo build --features sensitive-logs`

### Fixed
- 修复 SSE 流 ping 保活首次立即触发的问题
  - 使用 `interval_at(Instant::now() + ping_period, ping_period)` 延迟首次触发
  - 避免连接建立后立即发送无意义的 ping 事件

### Changed
- 改进服务启动错误处理
  - 绑定监听地址失败时输出错误日志并退出（exit code 1）
  - HTTP 服务异常退出时输出错误日志并退出（exit code 1）

### Fixed
- 修复合并上游后 `CredentialEntry` 结构体字段缺失导致的编译错误
  - 添加 `disable_reason: Option<DisableReason>` 字段（公共 API 展示用）
  - 添加 `auto_heal_reason: Option<AutoHealReason>` 字段（内部自愈逻辑用）
- 修复禁用原因字段不同步问题
  - `report_failure()`: 禁用时同步设置两个字段
  - `set_disabled()`: 启用/禁用时同步设置/清除两个字段
  - `reset_and_enable()`: 重置时同步清除两个字段
  - 自愈循环：重新启用凭据时同步清除 `disable_reason`
  - `mark_insufficient_balance()`: 清除 `auto_heal_reason` 防止被自愈循环错误恢复

### Changed
- 重命名内部字段以提高可读性
  - `DisabledReason` → `AutoHealReason`（自愈原因枚举）
  - `disabled_reason` → `auto_heal_reason`（自愈原因字段）
- 日志中的 `user_id` 现在会进行掩码处理，保护用户隐私
  - 长度 > 25：保留前13后8字符（如 `user_f516339a***897ac7`）
  - 长度 13-25：保留前4后4字符
  - 长度 ≤ 12：完全掩码为 `***`

### Added
- 新增缓存余额查询 API（`GET /credentials/balances/cached`）
  - 后端：`CachedBalanceInfo` 结构体、`get_all_cached_balances()` 方法
  - 前端：凭据卡片直接显示缓存余额和更新时间
  - 30 秒自动轮询更新，缓存超过 1 分钟时点击强制刷新
- 新增 Bonus 用量包支持（`src/kiro/model/usage_limits.rs`）
  - 新增 `Bonus` 结构体，支持 GIFT 类型的额外用量包
  - 新增 `Bonus::is_active()` 方法，按状态/过期时间判断是否激活
  - `usage_limit()` 和 `current_usage()` 现在会合并基础额度、免费试用额度和所有激活的 bonuses
- 新增 Kiro Web Portal API 模块（`src/kiro/web_portal.rs`）
  - 支持 CBOR 协议与 app.kiro.dev 通信
  - 实现 `get_user_info()` 和 `get_user_usage_and_limits()` API
  - 新增 `aggregate_account_info()` 聚合账号信息（套餐/用量/邮箱等）
- Admin UI 前端增强
  - 新增数字格式化工具（`admin-ui/src/lib/format.ts`）：K/M/B 显示、Token 对格式化、过期时间格式化
  - 新增统计相关 API 和 Hooks：`getCredentialStats`, `resetCredentialStats`, `resetAllStats`
  - 新增账号信息 API：`getCredentialAccountInfo`, `useCredentialAccountInfo`
  - 扩展 `CredentialStatusItem` 添加统计字段（调用次数、Token 用量、最后调用时间等）
  - 新增完整的账号信息类型定义（`AccountAggregateInfo`, `CreditsUsageSummary` 等）
- 新增 `serde_cbor` 依赖用于 CBOR 编解码

### Fixed
- 修复手动查询余额后列表页面不显示缓存余额的问题
  - `get_balance()` 成功后调用 `update_balance_cache()` 更新缓存
  - 现在点击"查看余额"后，列表页面会正确显示缓存的余额值
- 修复关闭余额弹窗后卡片不更新缓存余额的问题
  - 弹窗关闭时调用 `queryClient.invalidateQueries({ queryKey: ['cached-balances'] })`
  - 确保卡片和弹窗使用的两个独立数据源保持同步

### Changed
- 增强 Token 刷新日志，添加凭证 ID 追踪
  - 新增 `refresh_token_with_id()` 函数支持传入凭证 ID
  - 日志现在包含 `credential_id` 字段，便于多凭据环境下的问题排查

### Changed
- 调整重试策略：单凭据最大重试次数 3→2，单请求最大重试次数 9→5
  - `MAX_RETRIES_PER_CREDENTIAL`: 3 → 2
  - `MAX_TOTAL_RETRIES`: 9 → 5
  - `MAX_FAILURES_PER_CREDENTIAL`: 3 → 2
  - 减少无效凭据的重试开销，加快故障转移速度

### Added
- 新增用户亲和性绑定功能：连续对话优先使用同一凭据（基于 `metadata.user_id`）
  - 新增 `src/kiro/affinity.rs` 模块，实现 `UserAffinityManager`
  - 新增 `acquire_context_for_user()` 方法支持亲和性查询
  - 亲和性绑定 TTL 为 30 分钟
- 新增余额感知故障转移：凭据失效时自动切换到余额最高的可用凭据
- 新增动态余额缓存 TTL 策略：
  - 高频渠道（10分钟内 ≥20 次调用）：10 分钟刷新
  - 低频渠道：30 分钟刷新
  - 低余额渠道（余额 < 1.0）：24 小时刷新
- 新增 `record_usage()` 方法自动记录凭据使用频率
- 新增负载均衡：无亲和性绑定时优先分配到使用频率最低的凭据
- 新增 `DisableReason` 枚举，区分凭据禁用原因（失败次数、余额不足、模型不可用、手动禁用）
- 成功请求时自动重置 `MODEL_TEMPORARILY_UNAVAILABLE` 计数器，避免跨时间累计触发
- 新增 `MODEL_TEMPORARILY_UNAVAILABLE` 错误检测和全局禁用机制
  - 当该 500 错误发生 2 次时，自动禁用所有凭据
  - 5 分钟后自动恢复（余额不足的凭据除外）
- `CredentialEntrySnapshot` 新增 `disable_reason` 字段，支持查询禁用原因
- 新增自动余额刷新：成功请求后自动在后台刷新余额缓存（基于动态 TTL 策略）
  - 新增 `spawn_balance_refresh()` 方法，使用 `tokio::spawn` 异步刷新
  - 新增 `should_refresh_balance()` 方法，根据 TTL 判断是否需要刷新
