# Changelog

## [v1.0.17] - 2026-02-15

### Added
- **自适应压缩第四层：超长用户消息内容截断** (`src/anthropic/compressor.rs`, `src/anthropic/handlers.rs`)
  - 新增 `compress_long_messages_pass()` 函数，截断超长的 User 消息 content（保留头部，尾部附加省略标记）
  - 在 `adaptive_shrink_request_body` 的三层策略之后增加第四层兜底，解决单条消息过大（如粘贴整个文件）导致自适应压缩空转的问题
  - 动态计算截断阈值：初始为最大消息字符数的 3/4，每轮递减 3/4，最低 8192 字符
  - 日志新增 `final_message_content_max_chars` 字段便于排查

## [v1.0.16] - 2026-02-15

### Fixed
- **请求体日志仅在上游报错时输出完整内容** (`src/anthropic/handlers.rs`)
  - 移除发送前的完整请求体 DEBUG 日志（`sensitive-logs` 模式下每次请求都输出几十 KB JSON），统一只输出字节大小
  - 上游报错时在 `sensitive-logs` 模式下以 ERROR 级别输出完整请求体（截断 base64），用于诊断 400/502 等错误

## [v1.0.15] - 2026-02-15

### Added
- **Opus 4.6 1M 上下文窗口支持** (`src/anthropic/types.rs`, `src/anthropic/handlers.rs`, `src/anthropic/stream.rs`)
  - 新增 `get_context_window_size()` 函数，Opus 4.6 返回 1,000,000 tokens，其他模型返回 200,000 tokens
  - 删除硬编码 `CONTEXT_WINDOW_SIZE` 常量，改用动态计算
  - `MAX_BUDGET_TOKENS` 从 24,576 提升到 128,000
  - `Model` 结构体新增 `context_length`、`max_completion_tokens`、`thinking` 字段
- **Agentic 模型变体** (`src/anthropic/converter.rs`, `src/anthropic/handlers.rs`)
  - 新增 sonnet-agentic、opus-4.5-agentic、opus-4.6-agentic、haiku-agentic 四个模型变体
  - `map_model()` 自动剥离 `-agentic` 后缀再映射
  - Agentic 模型注入专用系统提示，引导自主工作模式
- **Thinking level 后缀** (`src/anthropic/handlers.rs`)
  - 支持 `-thinking-minimal`(512)、`-thinking-low`(1024)、`-thinking-medium`(8192)、`-thinking-high`(24576)、`-thinking-xhigh`(32768) 后缀
- **工具压缩** (`src/anthropic/tool_compression.rs` 新建)
  - 20KB 阈值两步压缩：简化 input_schema → 按比例截断 description（最短 50 字符）
- **截断检测** (`src/anthropic/truncation.rs` 新建)
  - 4 种截断类型的启发式检测（空输入、无效 JSON、缺少字段、未闭合字符串）
  - 工具 JSON 解析失败时自动检测截断并生成软失败消息

### Changed
- 工具调用仅含 tool_use 时占位符从 `" "` 改为 `"."`，提升语义清晰度

## [v1.0.14] - 2026-02-15

### Fixed
- **sensitive-logs 模式下请求体日志截断** (`src/kiro/provider.rs`, `src/anthropic/handlers.rs`)
  - 400 错误日志中的 `request_body` 字段改用 `truncate_body_for_log()` 截断（保留头尾各 1200 字符），避免输出包含大量 base64 图片数据的完整请求体
  - 工具输入 JSON 解析失败日志中的 `request_body` 字段改用 `truncate_middle()` 截断
  - 新增 `KiroProvider::truncate_body_for_log()` 函数，正确处理 UTF-8 多字节字符边界

## [v1.0.13] - 2026-02-14

### Fixed
- **请求体大小预检扣除图片 base64 字节** (`src/anthropic/handlers.rs`)
  - 新增 `total_image_bytes()` 函数，计算 KiroRequest 中所有图片 base64 数据的总字节数
  - 请求体大小阈值判断时扣除图片字节数，避免含图片的请求被误拒（图片在上游按像素块计 token，不应计入 body 大小阈值）
  - 自适应压缩循环中每轮重新计算图片字节（历史截断可能移除含图片的消息）
  - 错误提示信息增加 image_bytes 和 effective_bytes 字段，便于排查

## [v1.0.12] - 2026-02-14

### Fixed
- **WebSearch 仅纯搜索请求走本地处理** (`src/anthropic/websearch.rs`, `src/anthropic/handlers.rs`)
  - 新增 `should_handle_websearch_request()` 精确判断：仅当 tool_choice 强制选择 web_search、tools 仅含 web_search 单工具、或用户消息包含 `Perform a web search for the query:` 前缀时，才路由到本地 WebSearch 处理
  - 混合工具场景（web_search + 其他工具）改为剔除 web_search 后转发上游，避免普通对话被误当成搜索查询
  - 新增 `strip_web_search_tools()` 从 tools 列表中移除 web_search 工具
  - 搜索查询提取增加空白归一化处理

## [v1.0.11] - 2026-02-14

### Added
- **自适应二次压缩策略** (`src/anthropic/handlers.rs`)
  - 请求体超过 `max_request_body_bytes` 阈值时，自动迭代压缩：逐步降低 tool_result/tool_use_input 截断阈值，最后按轮移除最老历史消息
  - 最多迭代 32 轮，避免极端输入导致过长 CPU 消耗
  - `post_messages` 和 `post_messages_cc` 均支持自适应压缩
- **压缩后 tool_use/tool_result 配对修复** (`src/anthropic/compressor.rs`)
  - 新增 `repair_tool_pairing_pass()`：历史截断后自动移除孤立的 tool_use 和 tool_result
  - 解决截断破坏跨消息 tool_use→tool_result 配对导致上游返回 400 "Improperly formed request" 的问题
- **stable 版 `floor_char_boundary` 工具函数** (`src/common/utf8.rs`)
  - 新增 `common::utf8` 模块，提供 stable Rust 下的 `floor_char_boundary()` 实现
  - 统一替换项目中散落的 `str::floor_char_boundary()` nightly 调用

### Fixed
- **WebSearch 支持混合工具列表** (`src/anthropic/websearch.rs`)
  - `has_web_search_tool()` 改为只要 tools 中包含 web_search（按 name 或 type 判断）即走本地处理，不再要求 tools 仅有一个
  - `extract_search_query()` 改为取最后一条 user 消息，更符合多轮对话场景
  - 新增非流式（`stream: false`）响应支持，返回完整 JSON 而非 SSE 流

### Changed
- **迁移 `floor_char_boundary` 调用到 `common::utf8` 模块** (`src/anthropic/compressor.rs`, `src/anthropic/stream.rs`, `src/admin/service.rs`, `src/kiro/token_manager.rs`, `src/kiro/provider.rs`)
  - 移除各文件中重复的 `floor_char_boundary` 内联实现，统一使用 `crate::common::utf8::floor_char_boundary`

## [v1.0.10] - 2026-02-12

### Fixed
- **配额耗尽返回 429 而非 502** (`src/anthropic/handlers.rs`, `src/kiro/provider.rs`)
  - 所有凭据配额耗尽时返回 `429 Too Many Requests`（`rate_limit_error`），而非 `502 Bad Gateway`
  - 余额刷新时主动禁用低余额凭据（余额 < 1.0），402 分支同步清零余额缓存
- **亲和性检查不再触发限流** (`src/kiro/token_manager.rs`)
  - 亲和性检查改用 `check_rate_limit` 只读探测，消除"检查本身消耗速率配额"的恶性循环
  - 亲和性分流日志提升至 info 级别并脱敏 user_id，便于生产监控热点凭据

### Added
- **请求体大小预检** (`src/anthropic/handlers.rs`, `src/model/config.rs`)
  - 新增 `max_request_body_bytes` 配置项（默认 400KB），序列化后拦截超大请求避免无效上游往返
  - `post_messages` 和 `post_messages_cc` 均支持预检

### Changed
- **移除无意义的 max_tokens 调整逻辑** (`src/anthropic/handlers.rs`)
  - 删除 max_tokens 超限警告日志和调整逻辑，因为该值实际不传递给 Kiro 上游

## [v1.0.9] - 2026-02-12

### Fixed
- **修复 upstream 合并丢失的功能** (`src/anthropic/stream.rs`)
  - 恢复 `stop_reason` 优先级逻辑：高优先级原因可覆盖低优先级（model_context_window_exceeded > max_tokens > tool_use > end_turn）
  - 注释掉重复的 `content_block_start` 日志，避免日志噪音
  - 修复 `contextUsageEvent` 日志格式（保留4位小数）
  - 移除冗余的 `find_char_boundary` 函数，改用标准库 `str::floor_char_boundary()`

## [v1.0.8] - 2026-02-12

### Added
- **批量导入 token.json** (`src/admin/service.rs`, `src/admin/types.rs`)
  - 新增 `import_token_json` 接口，支持批量导入官方 token.json 格式凭据
  - 自动映射 provider/authMethod 字段，支持 dry-run 预览模式
  - 去重检测：通过 refreshToken 前缀匹配避免重复导入

- **缓存余额查询接口** (`src/admin/handlers.rs`, `src/admin/router.rs`)
  - 新增 `GET /admin/credentials/balances/cached` 端点
  - 返回所有凭据的缓存余额信息（含 TTL 和缓存时间）

### Changed
- **用户亲和性支持** (`src/kiro/token_manager.rs`, `src/kiro/provider.rs`)
  - 新增 `UserAffinityManager`：用户与凭据绑定，保持会话连续性
  - `acquire_context_for_user` 方法支持按 user_id 获取绑定凭据
  - 亲和性过期时间可配置，默认 30 分钟

- **余额缓存动态 TTL** (`src/kiro/token_manager.rs`)
  - 基于使用频率动态调整缓存 TTL（高频用户更短 TTL）
  - 新增 `update_balance_cache`、`get_all_cached_balances` 方法
  - 缓存持久化到 `kiro_balance_cache.json`

- **请求压缩管道** (`src/anthropic/converter.rs`, `src/anthropic/middleware.rs`)
  - `AppState` 新增 `compression_config` 字段
  - `convert_request` 支持 `CompressionConfig` 参数
  - 图片压缩、空白压缩、上下文截断等功能集成

- **凭据级代理配置** (`src/kiro/provider.rs`)
  - KiroProvider 支持按凭据动态选择代理
  - 新增 `get_client_for_credential` 方法，缓存凭据级 client
  - API 请求和 MCP 请求均使用凭据的 `effective_proxy()`

- **API Region 分离** (`src/kiro/provider.rs`)
  - `base_url`、`mcp_url`、`base_domain` 使用 `effective_api_region()`
  - 支持 Token 刷新和 API 调用使用不同 region

- **handlers 传递 user_id** (`src/anthropic/handlers.rs`)
  - 从请求 metadata 提取 user_id 并传递给 provider
  - 支持用户亲和性功能

### Fixed
- **修复 mask_user_id UTF-8 panic** (`src/anthropic/handlers.rs`)
  - 使用 `chars()` 按字符而非字节切片，避免多字节字符导致 panic

## [v1.0.7] - 2026-02-10

### Added
- **图片 Token 估算与压缩模块** (`src/image.rs`)
  - 新增 `estimate_image_tokens()`: 从 base64 数据解析图片尺寸并估算 token 数
  - 新增 `process_image()`: 根据配置对大图进行缩放压缩
  - 实现 Anthropic 官方公式: `tokens = (width × height) / 750`
  - 支持 JPEG/PNG/GIF/WebP 格式

### Changed
- **Token 计算支持图片** (`src/token.rs`)
  - `count_all_tokens_local()` 现在处理 `type: "image"` 的 ContentBlock
  - 调用 `estimate_image_tokens()` 计算图片 token，解决之前图片 token 被计为 0 的问题
- **协议转换支持图片压缩** (`src/anthropic/converter.rs`)
  - `process_message_content()` 新增图片压缩处理，根据配置自动缩放超限图片
  - 新增 `count_images_in_request()` 统计请求中图片总数，用于判断多图模式
  - 缩放后记录日志: `图片已缩放: (原始尺寸) -> (缩放后尺寸), tokens: xxx`
- **压缩配置新增图片参数** (`src/model/config.rs`)
  - `image_max_long_edge`: 长边最大像素，默认 1568（Anthropic 推荐值）
  - `image_max_pixels_single`: 单张图片最大总像素，默认 1,150,000（约 1600 tokens）
  - `image_max_pixels_multi`: 多图模式下单张最大像素，默认 4,000,000（2000×2000）
  - `image_multi_threshold`: 触发多图限制的图片数量阈值，默认 20

### Dependencies
- 新增 `image` crate (0.25): 图片处理（支持 JPEG/PNG/GIF/WebP）
- 新增 `base64` crate (0.22): Base64 编解码

## [v1.0.6] - 2026-02-10

### Changed
- **压缩统计日志改用字节单位** (`src/anthropic/handlers.rs`)
  - 移除不准确的 token 估算（`compressed_input_tokens`、`tokens_saved`），改为直接输出字节数
  - 字段重命名：`whitespace_saved` → `whitespace_bytes_saved` 等，明确单位语义
  - 注释更新：说明字节统计用于排查上游 ~400KB 请求体限制

### Added
- **日志脱敏工具模块** (`src/common/redact.rs`)
  - `redact_opt_string`: Option<String> 脱敏为存在性表示
  - `mask_email`: 邮箱脱敏（保留首字符）
  - `mask_aws_account_id_in_arn`: AWS ARN 中 account id 脱敏
  - `mask_url_userinfo`: URL userinfo 脱敏
  - `mask_user_agent_machine_id`: User-Agent 中 machine_id 脱敏

## [v1.0.5] - 2026-02-09

### Changed
- **请求日志输出压缩前后 token 估算** (`src/anthropic/converter.rs`, `src/anthropic/handlers.rs`)
  - `ConversionResult` 新增 `compression_stats` 字段，将压缩统计从 converter 内部返回给调用方
  - `post_messages` / `post_messages_cc` 在 `convert_request()` 之前估算 input tokens，`Received` 日志追加 `estimated_input_tokens`
  - 压缩完成后输出 token 对比日志：`estimated_input_tokens`、`compressed_input_tokens`、`tokens_saved` 及各项压缩明细
  - WebSearch 分支复用已有估算值，消除重复的 `count_all_tokens` 调用
- **`count_all_tokens` 改为接受借用** (`src/token.rs`)
  - 参数从按值传递改为引用（`&str`、`&[Message]`、`&Option<Vec<T>>`），消除 handlers 中的深拷贝开销
- **历史截断统计增加字节数** (`src/anthropic/compressor.rs`)
  - `CompressionStats` 新增 `history_bytes_saved` 字段，`total_saved()` 包含历史截断字节数
  - `compress_history_pass` 返回 `(turns_removed, bytes_saved)` 元组，token 估算准确计入历史截断

## [v1.0.4] - 2026-02-09

### Changed
- **Docker 构建优化**: 引入 `cargo-chef` 实现依赖层缓存，大幅加速重复构建
  - 新增 `planner` 阶段生成依赖 recipe，`builder` 阶段先编译依赖再编译源码
  - Docker 层缓存使得仅源码变更时无需重新编译所有依赖
- **GitHub Actions Docker 构建缓存**: 启用 GHA 缓存（`cache-from/cache-to: type=gha`）
- **Cargo.toml 优化**:
  - `lto` 从 `true`（fat LTO）改为 `"thin"`，加快编译速度同时保持较好的优化效果
  - `tokio` features 从 `full` 精简为实际使用的 5 个 feature（`rt-multi-thread`, `macros`, `net`, `time`, `sync`），减小编译体积

## [v1.0.3] - 2026-02-09

### Fixed
- **opus4-6 上下文超限错误识别** (`src/anthropic/handlers.rs`)
  - 将 `Improperly formed request` 纳入输入过长错误检测，返回 Claude Code 可识别的 `400 Input is too long` 而非 `502 Bad Gateway`
- **Opus 4.6 模型 ID 移除时间日期** (`src/anthropic/handlers.rs`)
  - `claude-opus-4-6-20260206` → `claude-opus-4-6`
- **post_messages 借用冲突编译错误** (`src/anthropic/handlers.rs`)
  - 将 `override_thinking_from_model_name` 调用移至 `user_id` 提取之前，修复 E0502 借用冲突

### Added
- **模型 "thinking" 后缀支持** (`src/anthropic/handlers.rs`, `src/anthropic/converter.rs`)
  - 模型名含 `-thinking` 后缀时自动覆写 thinking 配置（opus4.6 用 adaptive + high effort，其他用 enabled）
  - 模型列表新增 sonnet/opus4.5/opus4.6/haiku 的 thinking 变体

### Docs
- **README.md 补充输入压缩配置文档**
  - 新增"输入压缩"章节，说明 5 层压缩管道机制和 10 个可配置参数

## [v1.0.2] - 2026-02-09

### Fixed
- **opus4-6 上下文超限错误识别** (`src/anthropic/handlers.rs`)
  - 将 `Improperly formed request` 纳入输入过长错误检测，返回 Claude Code 可识别的 `400 Input is too long` 而非 `502 Bad Gateway`
- **Opus 4.6 模型 ID 移除时间日期** (`src/anthropic/handlers.rs`)
  - `claude-opus-4-6-20260206` → `claude-opus-4-6`
- **post_messages 借用冲突编译错误** (`src/anthropic/handlers.rs`)
  - 将 `override_thinking_from_model_name` 调用移至 `user_id` 提取之前，修复 E0502 借用冲突

### Added
- **模型 "thinking" 后缀支持** (`src/anthropic/handlers.rs`, `src/anthropic/converter.rs`)
  - 模型名含 `-thinking` 后缀时自动覆写 thinking 配置（opus4.6 用 adaptive + high effort，其他用 enabled）
  - 模型列表新增 sonnet/opus4.5/opus4.6/haiku 的 thinking 变体

### Docs
- **README.md 补充输入压缩配置文档**
  - 新增"输入压缩"章节，说明 5 层压缩管道机制和 10 个可配置参数

## [v1.0.1] - 2026-02-08

### Fixed
- **历史截断字符统计口径修正** (`src/anthropic/compressor.rs`)
  - `max_history_chars` 从按字节 `.len()` 改为按字符 `.chars().count()`，与配置语义一致
- **remove_thinking_blocks 不再全局 trim** (`src/anthropic/compressor.rs`)
  - 移除末尾 `.trim()`，避免意外吞掉原始内容的首尾空白
- **token.json 导入注释/逻辑统一** (`src/admin/service.rs`)
  - 更新注释：`builder-id → idc`（与实际映射一致）
  - 删除 `auth_method == "builder-id"` 死分支
- **Admin UI 验活开关 added=0 卡住修复** (`admin-ui/src/components/import-token-json-dialog.tsx`)
  - `enableVerify` 开启但无新增凭据时，直接跳转 result 步骤而非卡在 preview
- **工具描述截断 max_description_chars=0 语义修正** (`src/anthropic/converter.rs`)
  - `0` 现在表示"不截断"，而非截断为空字符串

### Added
- **输入压缩管道** (`src/anthropic/compressor.rs`)
  - 新增 5 层压缩管道，规避 Kiro 上游 ~400KB 请求体大小限制
  - 空白压缩：连续空行(3+)→2行，行尾空格移除，保留行首缩进
  - thinking 块处理：支持 discard/truncate/keep 三种策略
  - tool_result 智能截断：按行截断保留头尾，行数不足时回退字符级截断
  - tool_use input 截断：递归截断 JSON 值中的大字符串
  - 历史截断：保留系统消息对，按轮数/字符数从前往后成对移除
  - 12 个单元测试覆盖所有压缩层和边界条件
- **CompressionConfig 配置结构体** (`src/model/config.rs`)
  - 新增 `compression` 配置字段，支持通过 JSON 配置文件调整参数
  - 10 个可配置参数：总开关、空白压缩、thinking 策略、各截断阈值、历史限制
  - 工具描述截断阈值从硬编码 10000 改为可配置（默认 4000）
- **合并 upstream 新功能**: 从 upstream/master 拉取并融合大量新特性
  - **负载均衡模式** (`src/model/config.rs`, `src/kiro/token_manager.rs`, `src/admin/`)
    - 新增 `loadBalancingMode` 配置项，支持 `priority`（默认）和 `balanced`（Least-Used）两种模式
    - Admin API 新增 `GET/PUT /config/load-balancing` 端点
    - 前端新增负载均衡模式切换开关
  - **凭据统计与持久化** (`src/kiro/token_manager.rs`)
    - 新增 `success_count`、`last_used_at` 字段，跟踪每个凭据的调用统计
    - 新增 `save_stats_debounced` 防抖持久化机制，`Drop` 时自动保存
    - 新增 `refreshTokenHash` 字段用于前端重复检测
  - **前端批量操作** (`admin-ui/src/components/dashboard.tsx`)
    - 批量导入对话框 (`BatchImportDialog`)
    - 批量验活对话框 (`BatchVerifyDialog`)
    - 分页控件、批量选择/删除/恢复/验活功能
    - 凭据卡片新增 Checkbox 选择、email 显示、订阅等级、成功次数、剩余用量等信息
  - **配置文件路径管理** (`src/model/config.rs`)
    - `Config` 新增 `config_path` 字段和 `load()`/`save()` 方法，支持配置回写
  - **前端依赖**: 新增 `@radix-ui/react-checkbox` 组件

### Changed
- `convert_request()` 签名新增 `&CompressionConfig` 参数 (`src/anthropic/converter.rs`)
- `convert_tools()` 描述截断阈值参数化 (`src/anthropic/converter.rs`)
- `AppState` 新增 `compression_config` 字段 (`src/anthropic/middleware.rs`)
- `create_router_with_provider()` 新增 `CompressionConfig` 参数 (`src/anthropic/router.rs`)
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
- **[P0] API Key 日志泄露修复** (`src/main.rs`)
  - info 级别不再打印 API Key 前半段，仅显示末 4 位和长度
  - 完整前缀仅在 `sensitive-logs` feature 的 debug 级别输出
- **[P2] 占位工具大小写变体重复插入** (`src/anthropic/converter.rs`)
  - `collect_history_tool_names` 改为小写去重，避免 `read`/`Read` 等变体重复
  - 占位工具 push 后同步更新 `existing_tool_names` 集合
- **[P1] 统计与缓存写盘非原子操作** (`src/kiro/token_manager.rs`, `src/admin/service.rs`)
  - 统计数据和余额缓存改为临时文件 + 原子重命名，防止写入中断导致文件损坏
- **[P1] stop_reason 覆盖策略可能丢失信息** (`src/anthropic/stream.rs`)
  - `set_stop_reason()` 改为基于优先级覆盖，高优先级原因可覆盖低优先级原因
- **[P2] snapshot 重复计算 SHA-256** (`src/kiro/token_manager.rs`)
  - `CredentialEntry` 新增 `refresh_token_hash` 缓存字段
  - Token 刷新时自动更新哈希，`snapshot()` 优先使用缓存避免重复计算
- **Clippy 警告修复** (`src/model/config.rs`)
  - 修复 `field_reassign_with_default` 警告，改用结构体初始化语法
- **[P2] Assistant Prefill 静默丢弃** (`src/anthropic/converter.rs`, `src/anthropic/handlers.rs`)
  - 末尾 `assistant` 消息（prefill 场景）不再返回 400 错误，改为静默丢弃并回退到最后一条 `user` 消息
  - Claude 4.x 已弃用 assistant prefill，Kiro API 也不支持，转换器在入口处截断消息列表
  - 移除 `InvalidLastMessageRole` 错误变体，`build_history` 接受预处理后的消息切片
- **[P2] 凭据回写原子性** (`src/kiro/token_manager.rs`)
  - `persist_credentials` 改为临时文件 + `rename` 原子替换
  - 新增 `resolve_symlink_target` 辅助函数：优先 `canonicalize`，失败时用 `read_link` 解析 symlink
  - 保留原文件权限，防止 umask 导致凭据文件权限放宽
  - Windows 兼容：`rename` 前先删除已存在的目标文件
  - 避免进程崩溃或并发调用导致凭据文件损坏
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
- 优化 400 Bad Request "输入过长" 错误的日志输出 (`src/kiro/provider.rs`)
  - 对于 `CONTENT_LENGTH_EXCEEDS_THRESHOLD` / `Input is too long` 错误，不再输出完整请求体（太占空间且无调试价值）
  - 改为记录 `request_body_bytes`（字节数）和 `estimated_input_tokens`（估算 token 数）
  - 新增 `estimate_tokens()` 函数：基于 CJK/非 CJK 字符比例估算 token 数量
    - CJK 字符（中/日/韩）: token 数 = 字符数 / 1.5
    - 其他字符（英文等）: token 数 = 字符数 / 3.5
  - 新增 `is_input_too_long()` 和 `is_cjk_char()` 辅助函数

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
- `CredentialEntry` 结构体新增 `fingerprint` 字段，每个凭据独立生成设备指纹

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

- 修复测试代码使用 `serde_json::json!` 构造 Tool 对象导致的类型不匹配问题
  - 改用 `Tool` 结构体直接构造，确保类型安全
  - 涉及文件：`src/anthropic/websearch.rs`
- 修复 `select_best_candidate_id()` 中 NaN 余额处理问题
  - 在评分阶段将 NaN/Infinity 余额归一化为 0.0
  - 避免 NaN 被 `total_cmp` 视为最大值导致错误的凭据选择
  - 避免 NaN 导致 `scored` 被完全过滤后除零 panic
  - 涉及文件：`src/kiro/token_manager.rs`
- 新增 `system` 字段格式兼容性支持（`src/anthropic/types.rs`）
  - 支持字符串格式：`"system": "You are a helpful assistant"`（new-api 等网关添加的系统提示词）
  - 支持数组格式：`"system": [{"type": "text", "text": "..."}]`（Claude Code 原生格式）
  - 自动将字符串格式转换为单元素数组，保持内部处理一致性
  - 新增 6 个单元测试验证格式兼容性
- 新增请求体大小限制：50MB（`DefaultBodyLimit::max(50 * 1024 * 1024)`）
  - 涉及文件：`src/anthropic/router.rs`
- 调整全局禁用恢复时间：`GLOBAL_DISABLE_RECOVERY_MINUTES` 从 10 分钟降至 5 分钟
  - 加快模型暂时不可用后的自动恢复速度
- 调整总重试次数硬上限：`MAX_TOTAL_RETRIES` 从 5 降至 3
  - 进一步减少无效重试开销，加快故障转移速度
- 余额初始化改为顺序查询，每次间隔 0.5 秒避免触发限流
  - 从并发查询改为顺序查询（`initialize_balances()`）
  - 移除 30 秒整体超时机制
  - 涉及文件：`src/kiro/token_manager.rs`

- 修复 assistant 消息仅包含 tool_use 时 content 为空导致 Kiro API 报错的问题
  - 当 text_content 为空且存在 tool_uses 时，使用 "OK" 作为占位符
  - 涉及文件：`src/anthropic/converter.rs`
- 修复 `MODEL_TEMPORARILY_UNAVAILABLE` 错误检测逻辑未实际调用的问题
  - 在 `call_mcp()` 和 `call_api()` 中添加错误检测和熔断触发逻辑
  - 移除 `report_model_unavailable()` 和 `disable_all_credentials()` 的 `#[allow(dead_code)]` 标记
  - 现在当检测到该错误时会正确触发全局熔断机制

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
- 版本号升级至 2026.1.5
- TLS 库从 native-tls 切换至 rustls（reqwest 依赖调整）
- `authMethod` 自动推断：未指定时根据是否有 clientId/clientSecret 自动判断为 idc 或 social
- 移除 web_search/websearch 工具过滤（`is_unsupported_tool` 现在返回 false）

- 修复 machineId 格式兼容性问题，支持 UUID 格式自动转换为 64 字符十六进制
### Removed
- 移除 `current_id` 概念（后端和前端）
  - 后端：移除 `MultiTokenManager.current_id` 字段和相关方法（`switch_to_next`、`select_highest_priority`、`select_by_balance`、`credentials`）
  - 后端：移除 `ManagerSnapshot.current_id` 字段
  - 后端：移除 `CredentialStatusItem.is_current` 字段
  - 前端：移除 `CredentialsStatusResponse.currentId` 和 `CredentialStatusItem.isCurrent`
  - 原因：多用户并发访问时，"当前凭据"概念无意义，凭据选择由 `acquire_context_for_user()` 动态决定

- 新增启动时余额初始化功能
  - `initialize_balances()`: 启动时并发查询所有凭据余额并更新缓存
  - 整体超时 30 秒，避免阻塞启动流程
  - 初始化失败或超时时输出警告日志
- 改进凭据选择算法：从单一"使用次数最少"改为两级排序
  - 第一优先级：使用次数最少
  - 第二优先级：余额最多（使用次数相同时）
  - 未初始化余额的凭据会被降级处理，避免被优先选中
- 移除前端"当前活跃"凭据展示
  - 前端：移除凭据卡片的"当前"高亮和 Badge
  - 前端：移除 Dashboard 中的"当前活跃"统计卡片
  - 统计卡片布局从 3 列调整为 2 列

- 新增 `sensitive-logs` feature flag，显式启用才允许打印潜在敏感信息（仅用于排障）
  - 默认关闭：Kiro 请求体只输出长度，凭证只输出摘要信息
  - 启用方式：`cargo build --features sensitive-logs`

- 修复 SSE 流 ping 保活首次立即触发的问题
  - 使用 `interval_at(Instant::now() + ping_period, ping_period)` 延迟首次触发
  - 避免连接建立后立即发送无意义的 ping 事件
- 改进服务启动错误处理
  - 绑定监听地址失败时输出错误日志并退出（exit code 1）
  - HTTP 服务异常退出时输出错误日志并退出（exit code 1）

- 修复合并上游后 `CredentialEntry` 结构体字段缺失导致的编译错误
  - 添加 `disable_reason: Option<DisableReason>` 字段（公共 API 展示用）
  - 添加 `auto_heal_reason: Option<AutoHealReason>` 字段（内部自愈逻辑用）
- 修复禁用原因字段不同步问题
  - `report_failure()`: 禁用时同步设置两个字段
  - `set_disabled()`: 启用/禁用时同步设置/清除两个字段
  - `reset_and_enable()`: 重置时同步清除两个字段
  - 自愈循环：重新启用凭据时同步清除 `disable_reason`
  - `mark_insufficient_balance()`: 清除 `auto_heal_reason` 防止被自愈循环错误恢复
- 重命名内部字段以提高可读性
  - `DisabledReason` → `AutoHealReason`（自愈原因枚举）
  - `disabled_reason` → `auto_heal_reason`（自愈原因字段）
- 日志中的 `user_id` 现在会进行掩码处理，保护用户隐私
  - 长度 > 25：保留前13后8字符（如 `user_f516339a***897ac7`）
  - 长度 13-25：保留前4后4字符
  - 长度 ≤ 12：完全掩码为 `***`

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

- 修复手动查询余额后列表页面不显示缓存余额的问题
  - `get_balance()` 成功后调用 `update_balance_cache()` 更新缓存
  - 现在点击"查看余额"后，列表页面会正确显示缓存的余额值
- 修复关闭余额弹窗后卡片不更新缓存余额的问题
  - 弹窗关闭时调用 `queryClient.invalidateQueries({ queryKey: ['cached-balances'] })`
  - 确保卡片和弹窗使用的两个独立数据源保持同步
- 增强 Token 刷新日志，添加凭证 ID 追踪
  - 新增 `refresh_token_with_id()` 函数支持传入凭证 ID
  - 日志现在包含 `credential_id` 字段，便于多凭据环境下的问题排查
- 调整重试策略：单凭据最大重试次数 3→2，单请求最大重试次数 9→5
  - `MAX_RETRIES_PER_CREDENTIAL`: 3 → 2
  - `MAX_TOTAL_RETRIES`: 9 → 5
  - `MAX_FAILURES_PER_CREDENTIAL`: 3 → 2
  - 减少无效凭据的重试开销，加快故障转移速度

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
