# kiro-rs

一个用 Rust 编写的 Anthropic Claude API 兼容代理服务，将 Anthropic API 请求转换为 Kiro API 请求。

## 免责声明
本项目仅供研究使用, Use at your own risk, 使用本项目所导致的任何后果由使用人承担, 与本项目无关。
本项目与 AWS/KIRO/Anthropic/Claude 等官方无关, 本项目不代表官方立场。

## 注意！
因 TLS 默认从 native-tls 切换至 rustls，你可能需要专门安装证书后才能配置 HTTP 代理。可通过 `config.json` 的 `tlsBackend` 切回 `native-tls`。
如果遇到请求报错, 尤其是无法刷新 token, 或者是直接返回 error request, 请尝试切换 tls 后端为 `native-tls`, 一般即可解决。

**Write Failed/会话卡死**: 如果遇到持续的 Write File / Write Failed 并导致会话不可用，参考 Issue [#22](https://github.com/hank9999/kiro.rs/issues/22) 和 [#49](https://github.com/hank9999/kiro.rs/issues/49) 的说明与临时解决方案（通常与输出过长被截断有关，可尝试调低输出相关 token 上限）

## 功能特性

- **Anthropic API 兼容**: 完整支持 Anthropic Claude API 格式
- **流式响应**: 支持 SSE (Server-Sent Events) 流式输出
- **Token 自动刷新**: 自动管理和刷新 OAuth Token
- **多凭据支持**: 支持配置多个凭据，按优先级自动故障转移
- **智能重试**: 单凭据最多重试 2 次，单请求最多重试 5 次
- **凭据回写**: 多凭据格式下自动回写刷新后的 Token
- **Thinking 模式**: 支持 Claude 的 extended thinking 功能
- **工具调用**: 完整支持 function calling / tool use
- **多模型支持**: 支持 Sonnet、Opus、Haiku 系列模型

## 支持的 API 端点

### 标准端点 (/v1)

| 端点 | 方法 | 描述          |
|------|------|-------------|
| `/v1/models` | GET | 获取可用模型列表    |
| `/v1/messages` | POST | 创建消息（对话）    |
| `/v1/messages/count_tokens` | POST | 估算 Token 数量 |

### Claude Code 兼容端点 (/cc/v1)

| 端点 | 方法 | 描述          |
|------|------|-------------|
| `/cc/v1/messages` | POST | 创建消息（流式响应会等待上游完成后再返回，确保 `input_tokens` 准确） |
| `/cc/v1/messages/count_tokens` | POST | 估算 Token 数量（与 `/v1` 相同） |

> **`/cc/v1/messages` 与 `/v1/messages` 的区别**：
> - `/v1/messages`：实时流式返回，`message_start` 中的 `input_tokens` 是估算值
> - `/cc/v1/messages`：缓冲模式，等待上游流完成后，用从 `contextUsageEvent` 计算的准确 `input_tokens` 更正 `message_start`，然后一次性返回所有事件
> - 等待期间会每 25 秒发送 `ping` 事件保活

## 快速开始

> **前置步骤**：编译前需要先构建前端 Admin UI（用于嵌入到二进制中）：
> ```bash
> cd admin-ui && pnpm install && pnpm build
> ```

### 1. 编译项目

```bash
cargo build --release
```

### 2. 配置文件

在**当前工作目录**创建 `config.json` 配置文件（或通过 `-c` 参数指定路径）：

> ⚠️ **注意**：JSON 不支持注释，请勿复制带 `//` 注释的示例。下方提供可直接复制的配置。

**最小启动配置**（可直接复制使用）：

```json
{
  "apiKey": "sk-your-api-key"
}
```

> 其他字段均有默认值：`host` 默认 `127.0.0.1`，`port` 默认 `8080`，`region` 默认 `us-east-1`，`tlsBackend` 默认 `rustls`

**启用 Admin UI**（添加 `adminApiKey` 后可访问 `/admin` 管理界面）：

```json
{
  "apiKey": "sk-your-api-key",
  "adminApiKey": "sk-admin-your-key"
}
```

**推荐配置**（显式指定常用字段）：

```json
{
  "host": "127.0.0.1",
  "port": 8990,
  "apiKey": "sk-your-api-key",
  "region": "us-east-1"
}
```

**完整配置字段说明**：

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `apiKey` | ✅ | - | 请求鉴权 Token |
| `host` | ❌ | `127.0.0.1` | 监听地址 |
| `port` | ❌ | `8080` | 监听端口 |
| `region` | ❌ | `us-east-1` | AWS 区域 |
| `tlsBackend` | ❌ | `rustls` | TLS 后端：`"rustls"` 或 `"native-tls"` |
| `kiroVersion` | ❌ | `0.8.0` | Kiro IDE 版本，用于自定义请求特征 |
| `machineId` | ❌ | 自动生成 | 64 位机器码，用于自定义请求特征 |
| `systemVersion` | ❌ | 随机 | 系统版本标识，如 `"darwin#24.6.0"` |
| `nodeVersion` | ❌ | `22.21.1` | Node.js 版本标识 |
| `countTokensApiUrl` | ❌ | - | 外部 Token 统计 API 地址 |
| `countTokensApiKey` | ❌ | - | 外部 Token 统计 API 密钥 |
| `countTokensAuthType` | ❌ | `x-api-key` | 外部 API 认证类型：`"x-api-key"` 或 `"bearer"` |
| `proxyUrl` | ❌ | - | HTTP/SOCKS5 代理地址 |
| `proxyUsername` | ❌ | - | 代理用户名 |
| `proxyPassword` | ❌ | - | 代理密码 |
| `adminApiKey` | ❌ | - | Admin API 密钥，配置后启用 Web 管理界面 |
| `credentialRpm` | ❌ | - | 单凭据目标 RPM（每分钟请求数），用于凭据级节流分流 |

### 3. 凭证文件

在**当前工作目录**创建 `credentials.json` 凭证文件（或通过 `--credentials` 参数指定路径）。

凭证信息从 Kiro IDE 获取，支持两种格式：

#### 单凭据格式（旧格式，向后兼容）

**最小配置 - Social 登录**（可直接复制）：

```json
{
  "refreshToken": "你的刷新Token",
  "expiresAt": "2025-01-01T00:00:00.000Z",
  "authMethod": "social"
}
```

**最小配置 - IdC/Builder-ID/IAM 登录**（可直接复制）：

```json
{
  "refreshToken": "你的刷新Token",
  "expiresAt": "2025-01-01T00:00:00.000Z",
  "authMethod": "idc",
  "clientId": "你的clientId",
  "clientSecret": "你的clientSecret"
}
```

**单凭据字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| `refreshToken` | ✅ | 刷新 Token，有效期 7-30 天不等 |
| `expiresAt` | ✅ | Token 过期时间（RFC3339 格式），不确定可填已过期时间 |
| `authMethod` | ✅ | 认证方式：`"social"` 或 `"idc"`（IdC/Builder-ID/IAM 统一填 `"idc"`） |
| `accessToken` | ❌ | 访问 Token，可自动刷新 |
| `profileArn` | ❌ | AWS Profile ARN |
| `clientId` | ❌ | IdC 登录必填 |
| `clientSecret` | ❌ | IdC 登录必填 |

#### 多凭据格式（新格式，支持故障转移和自动回写）

```json
[
  {
    "refreshToken": "第一个凭据的刷新Token",
    "expiresAt": "2025-12-31T02:32:45.144Z",
    "authMethod": "social",
    "priority": 0
  },
  {
    "refreshToken": "第二个凭据的刷新Token",
    "expiresAt": "2025-12-31T02:32:45.144Z",
    "authMethod": "idc",
    "clientId": "xxxxxxxxx",
    "clientSecret": "xxxxxxxxx",
    "region": "us-east-2",
    "priority": 1
  }
]
```

> **多凭据特性说明**：
> - 按 `priority` 字段排序，数字越小优先级越高（默认为 0）
> - 单凭据最多重试 2 次，单请求最多重试 5 次
> - 自动故障转移到下一个可用凭据
> - 多凭据格式下 Token 刷新后自动回写到源文件
> - 可选的 `region` 字段：用于 OIDC token 刷新时指定 endpoint 区域，未配置时回退到 config.json 的 region
> - 可选的 `machineId` 字段：凭据级机器码；未配置时回退到 config.json 的 machineId；都未配置时由 refreshToken 派生

### 4. 启动服务

**方式一：默认路径启动**

将 `config.json` 和 `credentials.json` 放在当前工作目录下，直接运行：

```bash
./target/release/kiro-rs
# Windows: target\release\kiro-rs.exe
```

**方式二：指定配置文件路径**

```bash
./kiro-rs -c /path/to/config.json --credentials /path/to/credentials.json
```

**命令行参数**：

| 参数 | 说明 |
|------|------|
| `-c, --config` | 配置文件路径，默认为当前工作目录的 `config.json` |
| `--credentials` | 凭证文件路径，默认为当前工作目录的 `credentials.json` |
| `-h, --help` | 显示帮助信息 |
| `-V, --version` | 显示版本号 |

### 5. 使用 API

```bash
curl http://127.0.0.1:8990/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-your-custom-api-key" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

## 配置参考

> 详细字段说明见上方"快速开始"章节，此处仅列出类型和默认值。

### config.json

| 字段 | 类型 | 默认值 |
|------|------|--------|
| `apiKey` | string | - |
| `host` | string | `127.0.0.1` |
| `port` | number | `8080` |
| `region` | string | `us-east-1` |
| `tlsBackend` | string | `rustls` |
| `kiroVersion` | string | `0.8.0` |
| `machineId` | string | 自动生成 |
| `systemVersion` | string | 随机 |
| `nodeVersion` | string | `22.21.1` |
| `countTokensApiUrl` | string | - |
| `countTokensApiKey` | string | - |
| `countTokensAuthType` | string | `x-api-key` |
| `proxyUrl` | string | - |
| `proxyUsername` | string | - |
| `proxyPassword` | string | - |
| `adminApiKey` | string | - |
| `credentialRpm` | number | - |

### credentials.json

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | number | 凭据 ID（Admin API 用，手写可不填） |
| `accessToken` | string | 访问令牌（可自动刷新） |
| `refreshToken` | string | 刷新令牌 |
| `profileArn` | string | AWS Profile ARN |
| `expiresAt` | string | 过期时间（RFC3339） |
| `authMethod` | string | `social` 或 `idc` |
| `clientId` | string | IdC 登录必填 |
| `clientSecret` | string | IdC 登录必填 |
| `priority` | number | 优先级（多凭据时有效） |
| `region` | string | 凭据级 region |
| `machineId` | string | 凭据级机器码 |

> **说明**：IdC / Builder-ID / IAM 统一使用 `authMethod: "idc"`

## 模型映射

| Anthropic 模型 | Kiro 模型 |
|----------------|-----------|
| `*sonnet*` | `claude-sonnet-4.5` |
| `*opus*` | `claude-opus-4.5` |
| `*haiku*` | `claude-haiku-4.5` |

## 项目结构

```
kiro-rs/
├── src/
│   ├── main.rs                 # 程序入口
│   ├── model/                  # 配置和参数模型
│   │   ├── config.rs           # 应用配置
│   │   └── arg.rs              # 命令行参数
│   ├── anthropic/              # Anthropic API 兼容层
│   │   ├── router.rs           # 路由配置
│   │   ├── handlers.rs         # 请求处理器
│   │   ├── middleware.rs       # 认证中间件
│   │   ├── types.rs            # 类型定义
│   │   ├── converter.rs        # 协议转换器
│   │   ├── stream.rs           # 流式响应处理
│   │   └── token.rs            # Token 估算
│   └── kiro/                   # Kiro API 客户端
│       ├── provider.rs         # API 提供者
│       ├── token_manager.rs    # Token 管理
│       ├── machine_id.rs       # 设备指纹生成
│       ├── model/              # 数据模型
│       │   ├── credentials.rs  # OAuth 凭证
│       │   ├── events/         # 响应事件类型
│       │   ├── requests/       # 请求类型
│       │   └── common/         # 共享类型
│       └── parser/             # AWS Event Stream 解析器
│           ├── decoder.rs      # 流式解码器
│           ├── frame.rs        # 帧解析
│           ├── header.rs       # 头部解析
│           └── crc.rs          # CRC 校验
├── Cargo.toml                  # 项目配置
├── config.example.json         # 配置示例
├── admin-ui/                   # Admin UI 前端工程（构建产物会嵌入二进制）
├── tools/                      # 辅助工具
└── Dockerfile                  # Docker 构建文件
```

## 技术栈

- **Web 框架**: [Axum](https://github.com/tokio-rs/axum) 0.8
- **异步运行时**: [Tokio](https://tokio.rs/)
- **HTTP 客户端**: [Reqwest](https://github.com/seanmonstar/reqwest)
- **序列化**: [Serde](https://serde.rs/)
- **日志**: [tracing](https://github.com/tokio-rs/tracing)
- **命令行**: [Clap](https://github.com/clap-rs/clap)

## 高级功能

### Thinking 模式

支持 Claude 的 extended thinking 功能：

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  },
  "messages": [...]
}
```

### 工具调用

完整支持 Anthropic 的 tool use 功能：

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 1024,
  "tools": [
    {
      "name": "get_weather",
      "description": "获取指定城市的天气",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        },
        "required": ["city"]
      }
    }
  ],
  "messages": [...]
}
```

### 流式响应

设置 `stream: true` 启用 SSE 流式响应：

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 1024,
  "stream": true,
  "messages": [...]
}
```

## 认证方式

支持两种 API Key 认证方式：

1. **x-api-key Header**
   ```
   x-api-key: sk-your-api-key
   ```

2. **Authorization Bearer**
   ```
   Authorization: Bearer sk-your-api-key
   ```

## 环境变量

可通过环境变量配置日志级别：

```bash
RUST_LOG=debug ./target/release/kiro-rs
```

## 注意事项

1. **凭证安全**: 请妥善保管 `credentials.json` 文件，不要提交到版本控制
2. **Token 刷新**: 服务会自动刷新过期的 Token，无需手动干预
3. **WebSearch 工具**: 当 `tools` 列表仅包含一个 `web_search` 工具时，会走内置 WebSearch 转换逻辑

## Admin（可选）

当 `config.json` 配置了非空 `adminApiKey` 时，会启用：

- **Admin API（认证同 API Key）**
  - `GET /api/admin/credentials` - 获取所有凭据状态
  - `POST /api/admin/credentials` - 添加新凭据
  - `DELETE /api/admin/credentials/:id` - 删除凭据
  - `POST /api/admin/credentials/:id/disabled` - 设置凭据禁用状态
  - `POST /api/admin/credentials/:id/priority` - 设置凭据优先级
  - `POST /api/admin/credentials/:id/reset` - 重置失败计数
  - `GET /api/admin/credentials/:id/balance` - 获取凭据余额

- **Admin UI**
  - `GET /admin` - 访问管理页面（需要在编译前构建 `admin-ui/dist`）

## 💬 社区交流

欢迎加入 QQ 群交流讨论：**642217364**

<img src="docs/qrcode_1769645166806.png" width="300" alt="QQ群二维码">

## License

MIT

## 致谢

本项目的实现离不开前辈的努力:  
 - [kiro2api](https://github.com/caidaoli/kiro2api)
 - [proxycast](https://github.com/aiclientproxy/proxycast)

本项目部分逻辑参考了以上的项目, 再次由衷的感谢!
