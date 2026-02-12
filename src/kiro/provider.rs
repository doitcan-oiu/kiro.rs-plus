//! Kiro API Provider
//!
//! 核心组件，负责与 Kiro API 通信
//! 支持流式和非流式请求
//! 支持多凭据故障转移和重试

use parking_lot::Mutex;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONNECTION, CONTENT_TYPE, HOST, HeaderMap, HeaderValue};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use crate::http_client::{ProxyConfig, build_client};
use crate::kiro::machine_id;
use crate::kiro::token_manager::{CallContext, MultiTokenManager};

/// 每个凭据的最大重试次数
const MAX_RETRIES_PER_CREDENTIAL: usize = 2;

/// 总重试次数硬上限（避免无限重试）
const MAX_TOTAL_RETRIES: usize = 3;

/// Kiro API Provider
///
/// 核心组件，负责与 Kiro API 通信
/// 支持多凭据故障转移和重试机制
pub struct KiroProvider {
    token_manager: Arc<MultiTokenManager>,
    /// 默认 client（无代理或全局代理）
    default_client: Client,
    /// 全局代理配置
    global_proxy: Option<ProxyConfig>,
    /// 凭据级代理 client 缓存（key: credential_id）
    client_cache: Mutex<HashMap<u64, Client>>,
}

impl KiroProvider {
    /// 创建新的 KiroProvider 实例
    #[allow(dead_code)]
    pub fn new(token_manager: Arc<MultiTokenManager>) -> Self {
        Self::with_proxy(token_manager, None)
    }

    /// 创建带代理配置的 KiroProvider 实例
    pub fn with_proxy(token_manager: Arc<MultiTokenManager>, proxy: Option<ProxyConfig>) -> Self {
        let default_client = build_client(proxy.as_ref(), 720, token_manager.config().tls_backend)
            .expect("创建 HTTP 客户端失败");

        Self {
            token_manager,
            default_client,
            global_proxy: proxy,
            client_cache: Mutex::new(HashMap::new()),
        }
    }

    /// 获取凭据对应的 HTTP Client
    ///
    /// 优先使用凭据级代理，否则使用默认 client
    fn get_client_for_credential(&self, ctx: &CallContext) -> Client {
        let effective_proxy = ctx.credentials.effective_proxy(self.global_proxy.as_ref());

        // 如果凭据代理与全局代理相同，使用默认 client
        if effective_proxy == self.global_proxy {
            return self.default_client.clone();
        }

        // 检查缓存
        {
            let cache = self.client_cache.lock();
            if let Some(client) = cache.get(&ctx.id) {
                return client.clone();
            }
        }

        // 创建新 client 并缓存
        let client = build_client(
            effective_proxy.as_ref(),
            720,
            self.token_manager.config().tls_backend,
        )
        .unwrap_or_else(|e| {
            tracing::warn!("创建凭据级代理 client 失败，使用默认 client: {}", e);
            self.default_client.clone()
        });

        {
            let mut cache = self.client_cache.lock();
            cache.insert(ctx.id, client.clone());
        }

        client
    }

    /// 获取 token_manager 的引用
    #[allow(dead_code)]
    pub fn token_manager(&self) -> &MultiTokenManager {
        &self.token_manager
    }

    /// 获取 API 基础 URL（使用 effective_api_region）
    pub fn base_url(&self) -> String {
        format!(
            "https://q.{}.amazonaws.com/generateAssistantResponse",
            self.token_manager.config().effective_api_region()
        )
    }

    /// 获取 MCP API URL（使用 effective_api_region）
    pub fn mcp_url(&self) -> String {
        format!(
            "https://q.{}.amazonaws.com/mcp",
            self.token_manager.config().effective_api_region()
        )
    }

    /// 获取 API 基础域名（使用 effective_api_region）
    pub fn base_domain(&self) -> String {
        format!(
            "q.{}.amazonaws.com",
            self.token_manager.config().effective_api_region()
        )
    }

    /// 后台异步刷新余额缓存（如果需要）
    fn spawn_balance_refresh(&self, id: u64) {
        // 检查缓存是否需要刷新
        if !self.token_manager.should_refresh_balance(id) {
            return;
        }
        let tm = Arc::clone(&self.token_manager);
        tokio::spawn(async move {
            match tm.get_usage_limits_for(id).await {
                Ok(resp) => {
                    let remaining = resp.usage_limit() - resp.current_usage();
                    tm.update_balance_cache(id, remaining);
                    tracing::debug!("凭据 #{} 余额缓存已刷新: {:.2}", id, remaining);
                    if remaining < 1.0 {
                        tm.mark_insufficient_balance(id);
                        tracing::warn!("凭据 #{} 余额不足 ({:.2})，已主动禁用", id, remaining);
                    }
                }
                Err(e) => {
                    tracing::warn!("凭据 #{} 余额刷新失败: {}", id, e);
                }
            }
        });
    }

    /// 构建请求头
    ///
    /// # Arguments
    /// * `ctx` - API 调用上下文，包含凭据和 token
    fn build_headers(&self, ctx: &CallContext) -> anyhow::Result<HeaderMap> {
        let config = self.token_manager.config();

        let machine_id = machine_id::generate_from_credentials(&ctx.credentials, config)
            .ok_or_else(|| anyhow::anyhow!("无法生成 machine_id，请检查凭证配置"))?;

        let kiro_version = &config.kiro_version;
        let os_name = &config.system_version;
        let node_version = &config.node_version;

        let x_amz_user_agent = format!("aws-sdk-js/1.0.27 KiroIDE-{}-{}", kiro_version, machine_id);

        let user_agent = format!(
            "aws-sdk-js/1.0.27 ua/2.1 os/{} lang/js md/nodejs#{} api/codewhispererstreaming#1.0.27 m/E KiroIDE-{}-{}",
            os_name, node_version, kiro_version, machine_id
        );

        let mut headers = HeaderMap::new();

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-amzn-codewhisperer-optout",
            HeaderValue::from_static("true"),
        );
        headers.insert("x-amzn-kiro-agent-mode", HeaderValue::from_static("vibe"));
        headers.insert(
            "x-amz-user-agent",
            HeaderValue::from_str(&x_amz_user_agent)?,
        );
        headers.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_str(&user_agent)?,
        );
        headers.insert(HOST, HeaderValue::from_str(&self.base_domain())?);
        headers.insert(
            "amz-sdk-invocation-id",
            HeaderValue::from_str(&Uuid::new_v4().to_string())?,
        );
        headers.insert(
            "amz-sdk-request",
            HeaderValue::from_static("attempt=1; max=3"),
        );
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", ctx.token))?,
        );
        headers.insert(CONNECTION, HeaderValue::from_static("close"));

        Ok(headers)
    }

    /// 构建 MCP 请求头
    fn build_mcp_headers(&self, ctx: &CallContext) -> anyhow::Result<HeaderMap> {
        let config = self.token_manager.config();

        let machine_id = machine_id::generate_from_credentials(&ctx.credentials, config)
            .ok_or_else(|| anyhow::anyhow!("无法生成 machine_id，请检查凭证配置"))?;

        let kiro_version = &config.kiro_version;
        let os_name = &config.system_version;
        let node_version = &config.node_version;

        let x_amz_user_agent = format!("aws-sdk-js/1.0.27 KiroIDE-{}-{}", kiro_version, machine_id);

        let user_agent = format!(
            "aws-sdk-js/1.0.27 ua/2.1 os/{} lang/js md/nodejs#{} api/codewhispererstreaming#1.0.27 m/E KiroIDE-{}-{}",
            os_name, node_version, kiro_version, machine_id
        );

        let mut headers = HeaderMap::new();

        // 按照严格顺序添加请求头
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        headers.insert(
            "x-amz-user-agent",
            HeaderValue::from_str(&x_amz_user_agent)?,
        );
        headers.insert("user-agent", HeaderValue::from_str(&user_agent)?);
        headers.insert("host", HeaderValue::from_str(&self.base_domain())?);

        headers.insert(
            "amz-sdk-invocation-id",
            HeaderValue::from_str(&Uuid::new_v4().to_string())?,
        );
        headers.insert(
            "amz-sdk-request",
            HeaderValue::from_static("attempt=1; max=3"),
        );
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&format!("Bearer {}", ctx.token))?,
        );
        headers.insert("Connection", HeaderValue::from_static("close"));

        Ok(headers)
    }

    /// 发送非流式 API 请求
    ///
    /// 支持多凭据故障转移：
    /// - 400 Bad Request: 直接返回错误，不计入凭据失败
    /// - 401/403: 视为凭据/权限问题，计入失败次数并允许故障转移
    /// - 402 MONTHLY_REQUEST_COUNT: 视为额度用尽，禁用凭据并切换
    /// - 429/5xx/网络等瞬态错误: 重试但不禁用或切换凭据（避免误把所有凭据锁死）
    ///
    /// # Arguments
    /// * `request_body` - JSON 格式的请求体字符串
    ///
    /// # Returns
    /// 返回原始的 HTTP Response，不做解析
    pub async fn call_api(
        &self,
        request_body: &str,
        user_id: Option<&str>,
    ) -> anyhow::Result<reqwest::Response> {
        self.call_api_with_retry(request_body, false, user_id).await
    }

    /// 发送流式 API 请求
    ///
    /// 支持多凭据故障转移：
    /// - 400 Bad Request: 直接返回错误，不计入凭据失败
    /// - 401/403: 视为凭据/权限问题，计入失败次数并允许故障转移
    /// - 402 MONTHLY_REQUEST_COUNT: 视为额度用尽，禁用凭据并切换
    /// - 429/5xx/网络等瞬态错误: 重试但不禁用或切换凭据（避免误把所有凭据锁死）
    ///
    /// # Arguments
    /// * `request_body` - JSON 格式的请求体字符串
    ///
    /// # Returns
    /// 返回原始的 HTTP Response，调用方负责处理流式数据
    pub async fn call_api_stream(
        &self,
        request_body: &str,
        user_id: Option<&str>,
    ) -> anyhow::Result<reqwest::Response> {
        self.call_api_with_retry(request_body, true, user_id).await
    }

    /// 发送 MCP API 请求
    ///
    /// 用于 WebSearch 等工具调用
    ///
    /// # Arguments
    /// * `request_body` - JSON 格式的 MCP 请求体字符串
    ///
    /// # Returns
    /// 返回原始的 HTTP Response
    pub async fn call_mcp(&self, request_body: &str) -> anyhow::Result<reqwest::Response> {
        self.call_mcp_with_retry(request_body).await
    }

    /// 内部方法：带重试逻辑的 MCP API 调用
    async fn call_mcp_with_retry(&self, request_body: &str) -> anyhow::Result<reqwest::Response> {
        let total_credentials = self.token_manager.total_count();
        let max_retries = (total_credentials * MAX_RETRIES_PER_CREDENTIAL).min(MAX_TOTAL_RETRIES);
        let mut last_error: Option<anyhow::Error> = None;
        let mut forced_token_refresh: HashSet<u64> = HashSet::new();

        for attempt in 0..max_retries {
            // 获取调用上下文
            let ctx = match self.token_manager.acquire_context().await {
                Ok(c) => c,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };

            let url = self.mcp_url();
            let headers = match self.build_mcp_headers(&ctx) {
                Ok(h) => h,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };
            // 克隆 headers 用于错误日志（原 headers 会被 move）
            #[cfg(feature = "sensitive-logs")]
            let headers_for_log = headers.clone();

            // 获取凭据对应的 client（支持凭据级代理）
            let client = self.get_client_for_credential(&ctx);

            // 发送请求
            let response = match client
                .post(&url)
                .headers(headers)
                .body(request_body.to_string())
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::warn!(
                        "MCP 请求发送失败（尝试 {}/{}）: {}",
                        attempt + 1,
                        max_retries,
                        e
                    );
                    last_error = Some(e.into());
                    if attempt + 1 < max_retries {
                        sleep(Self::retry_delay(attempt)).await;
                    }
                    continue;
                }
            };

            let status = response.status();

            // 成功响应
            if status.is_success() {
                self.token_manager.report_success(ctx.id);
                return Ok(response);
            }

            // 失败响应
            let body = response.text().await.unwrap_or_default();

            // 402 额度用尽
            if status.as_u16() == 402 && Self::is_monthly_request_limit(&body) {
                let has_available = self.token_manager.report_quota_exhausted(ctx.id);
                if !has_available {
                    anyhow::bail!("MCP 请求失败（所有凭据已用尽）: {} {}", status, body);
                }
                last_error = Some(anyhow::anyhow!("MCP 请求失败: {} {}", status, body));
                continue;
            }

            // 400 Bad Request
            if status.as_u16() == 400 {
                let is_too_long = Self::is_input_too_long(&body);
                // 输入过长错误：只记录请求体大小，不输出完整内容（太占空间且无调试价值）
                if is_too_long {
                    let body_bytes = request_body.len();
                    let estimated_tokens = Self::estimate_tokens(request_body);
                    tracing::error!(
                        status = %status,
                        response_body_bytes = body.len(),
                        request_url = %url,
                        request_body_bytes = body_bytes,
                        estimated_input_tokens = estimated_tokens,
                        "MCP 400 Bad Request - 输入上下文过长"
                    );
                } else {
                    // 其他 400 错误：记录请求信息以便调试
                    #[cfg(feature = "sensitive-logs")]
                    tracing::error!(
                        status = %status,
                        response_body = %body,
                        request_url = %url,
                        request_headers = %Self::format_headers_for_log(&headers_for_log),
                        request_body = %request_body,
                        "MCP 400 Bad Request - 请求格式错误"
                    );
                    #[cfg(not(feature = "sensitive-logs"))]
                    tracing::error!(
                        status = %status,
                        response_body_bytes = body.len(),
                        request_url = %url,
                        request_body_bytes = request_body.len(),
                        "MCP 400 Bad Request - 请求格式错误"
                    );
                }
                #[cfg(feature = "sensitive-logs")]
                anyhow::bail!("MCP 请求失败: {} {}", status, body);
                #[cfg(not(feature = "sensitive-logs"))]
                {
                    if is_too_long {
                        let body_bytes = request_body.len();
                        let estimated_tokens = Self::estimate_tokens(request_body);
                        anyhow::bail!(
                            "MCP 请求失败: {} Input is too long. (request_body_bytes={}, estimated_input_tokens={})",
                            status,
                            body_bytes,
                            estimated_tokens
                        );
                    }

                    let summary = Self::summarize_error_body(&body);
                    anyhow::bail!("MCP 请求失败: {} {}", status, summary);
                }
            }

            // 401/403 凭据问题
            if matches!(status.as_u16(), 401 | 403) {
                // bearer token 失效：优先触发刷新再重试（避免因 expiresAt 不准导致误判/误禁用）
                if Self::is_invalid_bearer_token(&body) && forced_token_refresh.insert(ctx.id) {
                    tracing::warn!(
                        "MCP 请求失败（Bearer token 无效，触发刷新后重试，尝试 {}/{}）: {} {}",
                        attempt + 1,
                        max_retries,
                        status,
                        body
                    );
                    self.token_manager.invalidate_access_token(ctx.id);
                    last_error = Some(anyhow::anyhow!("MCP 请求失败: {} {}", status, body));
                    continue;
                }

                let has_available = self.token_manager.report_failure(ctx.id);
                if !has_available {
                    anyhow::bail!("MCP 请求失败（所有凭据已用尽）: {} {}", status, body);
                }
                last_error = Some(anyhow::anyhow!("MCP 请求失败: {} {}", status, body));
                continue;
            }

            // 瞬态错误
            if matches!(status.as_u16(), 408 | 429) || status.is_server_error() {
                tracing::warn!(
                    "MCP 请求失败（上游瞬态错误，尝试 {}/{}）: {} {}",
                    attempt + 1,
                    max_retries,
                    status,
                    body
                );

                // 检测 MODEL_TEMPORARILY_UNAVAILABLE 并触发熔断机制
                if Self::is_model_temporarily_unavailable(&body)
                    && self.token_manager.report_model_unavailable()
                {
                    // 熔断已触发，所有凭据已禁用，立即返回错误
                    anyhow::bail!(
                        "MCP 请求失败（模型暂时不可用，已触发熔断）: {} {}",
                        status,
                        body
                    );
                }

                last_error = Some(anyhow::anyhow!("MCP 请求失败: {} {}", status, body));
                if attempt + 1 < max_retries {
                    sleep(Self::retry_delay(attempt)).await;
                }
                continue;
            }

            // 其他 4xx
            if status.is_client_error() {
                anyhow::bail!("MCP 请求失败: {} {}", status, body);
            }

            // 兜底
            last_error = Some(anyhow::anyhow!("MCP 请求失败: {} {}", status, body));
            if attempt + 1 < max_retries {
                sleep(Self::retry_delay(attempt)).await;
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!("MCP 请求失败：已达到最大重试次数（{}次）", max_retries)
        }))
    }

    /// 内部方法：带重试逻辑的 API 调用
    ///
    /// 重试策略：
    /// - 每个凭据最多重试 MAX_RETRIES_PER_CREDENTIAL 次
    /// - 总重试次数 = min(凭据数量 × 每凭据重试次数, MAX_TOTAL_RETRIES)
    /// - 硬上限 3 次，避免无限重试
    async fn call_api_with_retry(
        &self,
        request_body: &str,
        is_stream: bool,
        user_id: Option<&str>,
    ) -> anyhow::Result<reqwest::Response> {
        let total_credentials = self.token_manager.total_count();
        let max_retries = (total_credentials * MAX_RETRIES_PER_CREDENTIAL).min(MAX_TOTAL_RETRIES);
        let mut last_error: Option<anyhow::Error> = None;
        let mut forced_token_refresh: HashSet<u64> = HashSet::new();
        let api_type = if is_stream { "流式" } else { "非流式" };

        for attempt in 0..max_retries {
            // 获取调用上下文（绑定 index、credentials、token），支持用户亲和性
            let ctx = match self.token_manager.acquire_context_for_user(user_id).await {
                Ok(c) => c,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };

            let url = self.base_url();
            let headers = match self.build_headers(&ctx) {
                Ok(h) => h,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };
            // 克隆 headers 用于错误日志（原 headers 会被 move）
            #[cfg(feature = "sensitive-logs")]
            let headers_for_log = headers.clone();

            // 动态注入当前凭据的 profile_arn（修复 IDC 凭据 403 问题）
            // IDC 凭据的 Token 刷新不返回 profile_arn，需要使用凭据自身的 profile_arn
            let final_body = match Self::inject_profile_arn(request_body, &ctx.credentials) {
                Ok(body) => body,
                Err(e) => {
                    tracing::warn!("注入 profile_arn 失败，使用原始请求体: {}", e);
                    request_body.to_string()
                }
            };
            // 克隆 final_body 用于错误日志（原 final_body 会被 move 到 body()）
            let final_body_for_log = final_body.clone();

            // 获取凭据对应的 client（支持凭据级代理）
            let client = self.get_client_for_credential(&ctx);

            // 发送请求
            let response = match client
                .post(&url)
                .headers(headers)
                .body(final_body)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::warn!(
                        "API 请求发送失败（尝试 {}/{}）: {}",
                        attempt + 1,
                        max_retries,
                        e
                    );
                    // 网络错误通常是上游/链路瞬态问题，不应导致"禁用凭据"或"切换凭据"
                    // （否则一段时间网络抖动会把所有凭据都误禁用，需要重启才能恢复）
                    last_error = Some(e.into());
                    if attempt + 1 < max_retries {
                        sleep(Self::retry_delay(attempt)).await;
                    }
                    continue;
                }
            };

            let status = response.status();

            // 成功响应
            if status.is_success() {
                self.token_manager.report_success(ctx.id);
                tracing::info!(credential_id = %ctx.id, "API 请求成功");
                // 后台异步刷新余额缓存
                self.spawn_balance_refresh(ctx.id);
                return Ok(response);
            }

            // 失败响应：读取 body 用于日志/错误信息
            let body = response.text().await.unwrap_or_default();

            // 402 Payment Required 且额度用尽：禁用凭据并故障转移
            if status.as_u16() == 402 && Self::is_monthly_request_limit(&body) {
                tracing::warn!(
                    "API 请求失败（额度已用尽，禁用凭据并切换，尝试 {}/{}）: {} {}",
                    attempt + 1,
                    max_retries,
                    status,
                    body
                );

                let has_available = self.token_manager.report_quota_exhausted(ctx.id);
                self.token_manager.update_balance_cache(ctx.id, 0.0);
                if !has_available {
                    anyhow::bail!(
                        "{} API 请求失败（所有凭据已用尽）: {} {}",
                        api_type,
                        status,
                        body
                    );
                }

                last_error = Some(anyhow::anyhow!(
                    "{} API 请求失败: {} {}",
                    api_type,
                    status,
                    body
                ));
                continue;
            }

            // 400 Bad Request - 请求问题，重试/切换凭据无意义
            if status.as_u16() == 400 {
                let is_too_long = Self::is_input_too_long(&body);
                // 输入过长错误：只记录请求体大小，不输出完整内容（太占空间且无调试价值）
                if is_too_long {
                    let body_bytes = final_body_for_log.len();
                    let estimated_tokens = Self::estimate_tokens(&final_body_for_log);
                    tracing::error!(
                        status = %status,
                        response_body_bytes = body.len(),
                        request_url = %url,
                        request_body_bytes = body_bytes,
                        estimated_input_tokens = estimated_tokens,
                        "400 Bad Request - 输入上下文过长"
                    );
                } else {
                    // 其他 400 错误：记录请求信息以便调试
                    #[cfg(feature = "sensitive-logs")]
                    tracing::error!(
                        status = %status,
                        response_body = %body,
                        request_url = %url,
                        request_headers = %Self::format_headers_for_log(&headers_for_log),
                        request_body = %final_body_for_log,
                        "400 Bad Request - 请求格式错误"
                    );
                    #[cfg(not(feature = "sensitive-logs"))]
                    tracing::error!(
                        status = %status,
                        response_body_bytes = body.len(),
                        request_url = %url,
                        request_body_bytes = final_body_for_log.len(),
                        "400 Bad Request - 请求格式错误"
                    );
                }
                #[cfg(feature = "sensitive-logs")]
                anyhow::bail!("{} API 请求失败: {} {}", api_type, status, body);
                #[cfg(not(feature = "sensitive-logs"))]
                {
                    // 对用户保留可区分的错误信息（例如 Input is too long），但避免返回过长内容。
                    if is_too_long {
                        let body_bytes = final_body_for_log.len();
                        let estimated_tokens = Self::estimate_tokens(&final_body_for_log);
                        anyhow::bail!(
                            "{} API 请求失败: {} Input is too long. (request_body_bytes={}, estimated_input_tokens={})",
                            api_type,
                            status,
                            body_bytes,
                            estimated_tokens
                        );
                    }

                    let summary = Self::summarize_error_body(&body);
                    anyhow::bail!("{} API 请求失败: {} {}", api_type, status, summary);
                }
            }

            // 401/403 - 更可能是凭据/权限问题：计入失败并允许故障转移
            if matches!(status.as_u16(), 401 | 403) {
                // bearer token 失效：优先触发刷新再重试（避免因 expiresAt 不准导致误判/误禁用）
                if Self::is_invalid_bearer_token(&body) && forced_token_refresh.insert(ctx.id) {
                    tracing::warn!(
                        "API 请求失败（Bearer token 无效，触发刷新后重试，尝试 {}/{}）: {} {}",
                        attempt + 1,
                        max_retries,
                        status,
                        body
                    );
                    self.token_manager.invalidate_access_token(ctx.id);
                    last_error = Some(anyhow::anyhow!(
                        "{} API 请求失败: {} {}",
                        api_type,
                        status,
                        body
                    ));
                    continue;
                }

                tracing::warn!(
                    "API 请求失败（可能为凭据错误，尝试 {}/{}）: {} {}",
                    attempt + 1,
                    max_retries,
                    status,
                    body
                );

                let has_available = self.token_manager.report_failure(ctx.id);
                if !has_available {
                    anyhow::bail!(
                        "{} API 请求失败（所有凭据已用尽）: {} {}",
                        api_type,
                        status,
                        body
                    );
                }

                last_error = Some(anyhow::anyhow!(
                    "{} API 请求失败: {} {}",
                    api_type,
                    status,
                    body
                ));
                continue;
            }

            // 429/408/5xx - 瞬态上游错误：重试但不禁用或切换凭据
            // （避免 429 high traffic / 502 high load 等瞬态错误把所有凭据锁死）
            if matches!(status.as_u16(), 408 | 429) || status.is_server_error() {
                tracing::warn!(
                    "API 请求失败（上游瞬态错误，尝试 {}/{}）: {} {}",
                    attempt + 1,
                    max_retries,
                    status,
                    body
                );
                last_error = Some(anyhow::anyhow!(
                    "{} API 请求失败: {} {}",
                    api_type,
                    status,
                    body
                ));
                if attempt + 1 < max_retries {
                    sleep(Self::retry_delay(attempt)).await;
                }
                continue;
            }

            // 其他 4xx - 通常为请求/配置问题：直接返回，不计入凭据失败
            if status.is_client_error() {
                anyhow::bail!("{} API 请求失败: {} {}", api_type, status, body);
            }

            // 兜底：当作可重试的瞬态错误处理（不切换凭据）
            tracing::warn!(
                "API 请求失败（未知错误，尝试 {}/{}）: {} {}",
                attempt + 1,
                max_retries,
                status,
                body
            );
            last_error = Some(anyhow::anyhow!(
                "{} API 请求失败: {} {}",
                api_type,
                status,
                body
            ));
            if attempt + 1 < max_retries {
                sleep(Self::retry_delay(attempt)).await;
            }
        }

        // 所有重试都失败
        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "{} API 请求失败：已达到最大重试次数（{}次）",
                api_type,
                max_retries
            )
        }))
    }

    /// 根据认证方式处理请求体中的 profileArn 字段
    ///
    /// 参考 CLIProxyAPIPlus 的 getEffectiveProfileArn 实现：
    /// - AWS SSO OIDC (Builder ID/IDC) 用户**不需要** profileArn，发送它会导致 403 错误
    /// - 只有 Kiro Desktop (social auth) 用户需要 profileArn
    ///
    /// # 行为
    /// - 如果是 builder-id 或 idc 认证：移除请求体中的 profileArn 字段
    /// - 如果是 social 认证且凭据有 profile_arn：注入/覆盖 profileArn 字段
    /// - 其他情况：保持请求体不变
    fn inject_profile_arn(
        request_body: &str,
        credentials: &crate::kiro::model::credentials::KiroCredentials,
    ) -> anyhow::Result<String> {
        // 检查认证方式
        let auth_method = credentials.auth_method.as_deref();

        // AWS SSO OIDC (Builder ID/IDC) - 不需要 profileArn，发送会导致 403
        // 同时检查 client_id + client_secret 的存在（AWS SSO OIDC 的特征）
        let is_aws_sso_oidc = matches!(auth_method, Some("builder-id") | Some("idc"))
            || (credentials.client_id.is_some() && credentials.client_secret.is_some());

        if is_aws_sso_oidc {
            // 解析请求体，移除 profileArn 字段
            let mut request: serde_json::Value = serde_json::from_str(request_body)?;

            if let Some(obj) = request.as_object_mut()
                && obj.remove("profileArn").is_some()
            {
                tracing::debug!(
                    "已移除 profileArn 字段（auth_method={:?}，AWS SSO OIDC 不需要）",
                    auth_method
                );
            }

            return Ok(serde_json::to_string(&request)?);
        }

        // Social auth - 需要 profileArn
        // 凭据没有 profile_arn 时，保持请求体不变
        let Some(profile_arn) = &credentials.profile_arn else {
            return Ok(request_body.to_string());
        };

        // 解析请求体为 JSON
        let mut request: serde_json::Value = serde_json::from_str(request_body)?;

        // 安全检查：确保是对象类型，避免在非对象 JSON 上 panic
        let obj = request
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("request body is not a JSON object"))?;

        // 注入 profile_arn（覆盖原有值）
        obj.insert(
            "profileArn".to_string(),
            serde_json::Value::String(profile_arn.clone()),
        );

        // 序列化回字符串
        Ok(serde_json::to_string(&request)?)
    }

    fn retry_delay(attempt: usize) -> Duration {
        // 指数退避 + 少量抖动，避免上游抖动时放大故障
        const BASE_MS: u64 = 200;
        const MAX_MS: u64 = 2_000;
        let exp = BASE_MS.saturating_mul(2u64.saturating_pow(attempt.min(6) as u32));
        let backoff = exp.min(MAX_MS);
        let jitter_max = (backoff / 4).max(1);
        let jitter = fastrand::u64(0..=jitter_max);
        Duration::from_millis(backoff.saturating_add(jitter))
    }

    fn is_monthly_request_limit(body: &str) -> bool {
        if body.contains("MONTHLY_REQUEST_COUNT") {
            return true;
        }

        let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
            return false;
        };

        if value
            .get("reason")
            .and_then(|v| v.as_str())
            .is_some_and(|v| v == "MONTHLY_REQUEST_COUNT")
        {
            return true;
        }

        value
            .pointer("/error/reason")
            .and_then(|v| v.as_str())
            .is_some_and(|v| v == "MONTHLY_REQUEST_COUNT")
    }

    /// 检测是否为 MODEL_TEMPORARILY_UNAVAILABLE 错误
    fn is_model_temporarily_unavailable(body: &str) -> bool {
        if body.contains("MODEL_TEMPORARILY_UNAVAILABLE") {
            return true;
        }

        let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
            return false;
        };

        if value
            .get("reason")
            .and_then(|v| v.as_str())
            .is_some_and(|v| v == "MODEL_TEMPORARILY_UNAVAILABLE")
        {
            return true;
        }

        value
            .pointer("/error/reason")
            .and_then(|v| v.as_str())
            .is_some_and(|v| v == "MODEL_TEMPORARILY_UNAVAILABLE")
    }

    /// 检测是否为「bearer token invalid」类错误
    ///
    /// 典型返回：
    /// `{"message":"The bearer token included in the request is invalid.","reason":null}`
    fn is_invalid_bearer_token(body: &str) -> bool {
        let lower = body.to_ascii_lowercase();
        lower.contains("bearer token") && lower.contains("invalid")
    }

    /// 检测是否为「输入过长」类错误
    ///
    /// 典型返回：
    /// `{"message":"Input is too long.","reason":"CONTENT_LENGTH_EXCEEDS_THRESHOLD"}`
    fn is_input_too_long(body: &str) -> bool {
        body.contains("CONTENT_LENGTH_EXCEEDS_THRESHOLD") || body.contains("Input is too long")
    }

    /// 从上游响应体提取一个适合返回给客户端的错误摘要
    ///
    /// 目标：
    /// - 保留关键错误信息（例如 "Input is too long" / "Improperly formed request"）
    /// - 避免返回过长/不可控的内容导致客户端难以区分或处理
    #[cfg(not(feature = "sensitive-logs"))]
    fn summarize_error_body(body: &str) -> String {
        const MAX_LEN: usize = 256;
        let trimmed = body.trim();
        if trimmed.is_empty() {
            return "<empty response body>".to_string();
        }

        // 优先尝试解析 JSON，从常见字段中提取 message / reason。
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            let message = value
                .get("message")
                .and_then(|v| v.as_str())
                .or_else(|| value.get("Message").and_then(|v| v.as_str()))
                .or_else(|| value.pointer("/error/message").and_then(|v| v.as_str()))
                .or_else(|| value.pointer("/error/Message").and_then(|v| v.as_str()));

            let reason = value
                .get("reason")
                .and_then(|v| v.as_str())
                .or_else(|| value.get("Reason").and_then(|v| v.as_str()))
                .or_else(|| value.pointer("/error/reason").and_then(|v| v.as_str()))
                .or_else(|| value.pointer("/error/Reason").and_then(|v| v.as_str()));

            if let Some(msg) = message {
                let mut s = msg.to_string();
                if let Some(r) = reason.filter(|r| !r.is_empty() && *r != "null") {
                    // 避免重复拼接（有些上游会把 reason 直接写入 message）
                    if !msg.contains(r) {
                        s.push_str(&format!(" (reason={})", r));
                    }
                }
                return Self::truncate_one_line(&s, MAX_LEN);
            }
        }

        // JSON 解析失败或不含常见字段，回退到压缩后的纯文本。
        Self::truncate_one_line(trimmed, MAX_LEN)
    }

    #[cfg(not(feature = "sensitive-logs"))]
    fn truncate_one_line(s: &str, max_len: usize) -> String {
        let one_line = s.split_whitespace().collect::<Vec<_>>().join(" ");
        if one_line.len() <= max_len {
            return one_line;
        }

        let end = one_line.floor_char_boundary(max_len);
        format!("{}...", &one_line[..end])
    }

    /// 估算文本的 token 数量
    ///
    /// 基于字符类型的估算公式：
    /// - CJK 字符（中/日/韩）: token 数 = 字符数 / 1.5
    /// - 其他字符（英文等）: token 数 = 字符数 / 3.5
    fn estimate_tokens(text: &str) -> usize {
        let mut cjk_count = 0usize;
        let mut other_count = 0usize;

        for c in text.chars() {
            if Self::is_cjk_char(c) {
                cjk_count += 1;
            } else {
                other_count += 1;
            }
        }

        let cjk_tokens = cjk_count as f64 / 1.5;
        let other_tokens = other_count as f64 / 3.5;
        (cjk_tokens + other_tokens + 0.5) as usize
    }

    /// 判断是否为 CJK（中日韩）字符
    #[inline]
    fn is_cjk_char(c: char) -> bool {
        matches!(c,
            '\u{4E00}'..='\u{9FFF}'   |  // CJK 统一汉字
            '\u{3400}'..='\u{4DBF}'   |  // CJK 扩展 A
            '\u{20000}'..='\u{2A6DF}' |  // CJK 扩展 B
            '\u{2A700}'..='\u{2B73F}' |  // CJK 扩展 C
            '\u{2B740}'..='\u{2B81F}' |  // CJK 扩展 D
            '\u{F900}'..='\u{FAFF}'   |  // CJK 兼容汉字
            '\u{2F800}'..='\u{2FA1F}' |  // CJK 兼容扩展
            '\u{3000}'..='\u{303F}'   |  // CJK 标点符号
            '\u{3040}'..='\u{309F}'   |  // 平假名
            '\u{30A0}'..='\u{30FF}'   |  // 片假名
            '\u{AC00}'..='\u{D7AF}'      // 韩文音节
        )
    }

    /// 格式化 HeaderMap 为可读字符串（用于日志输出）
    /// 敏感头部（Authorization）会被脱敏处理
    #[cfg(feature = "sensitive-logs")]
    fn format_headers_for_log(headers: &HeaderMap) -> String {
        headers
            .iter()
            .map(|(name, value)| {
                let value_str = value.to_str().unwrap_or("<binary>");
                // 脱敏 Authorization 头
                if name.as_str().eq_ignore_ascii_case("authorization") {
                    let masked = if value_str.len() > 20 {
                        format!(
                            "{}...{}",
                            &value_str[..10],
                            &value_str[value_str.len() - 6..]
                        )
                    } else {
                        "***".to_string()
                    };
                    format!("{}: {}", name, masked)
                } else {
                    format!("{}: {}", name, value_str)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::kiro::model::credentials::KiroCredentials;
    use crate::kiro::token_manager::CallContext;
    use crate::model::config::Config;

    fn create_test_provider(config: Config, credentials: KiroCredentials) -> KiroProvider {
        let tm = MultiTokenManager::new(config, vec![credentials], None, None, false).unwrap();
        KiroProvider::new(Arc::new(tm))
    }

    #[test]
    fn test_base_url() {
        let config = Config::default();
        let credentials = KiroCredentials::default();
        let provider = create_test_provider(config, credentials);
        assert!(provider.base_url().contains("amazonaws.com"));
        assert!(provider.base_url().contains("generateAssistantResponse"));
    }

    #[test]
    fn test_base_domain() {
        let mut config = Config::default();
        config.region = "us-east-1".to_string();
        let credentials = KiroCredentials::default();
        let provider = create_test_provider(config, credentials);
        assert_eq!(provider.base_domain(), "q.us-east-1.amazonaws.com");
    }

    #[test]
    fn test_build_headers() {
        let mut config = Config::default();
        config.region = "us-east-1".to_string();
        config.kiro_version = "0.8.0".to_string();

        let mut credentials = KiroCredentials::default();
        credentials.profile_arn = Some("arn:aws:sso::123456789:profile/test".to_string());
        credentials.refresh_token = Some("a".repeat(150));

        let provider = create_test_provider(config, credentials.clone());
        let ctx = CallContext {
            id: 1,
            credentials,
            token: "test_token".to_string(),
        };
        let headers = provider.build_headers(&ctx).unwrap();

        assert_eq!(headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert_eq!(headers.get("x-amzn-codewhisperer-optout").unwrap(), "true");
        assert_eq!(headers.get("x-amzn-kiro-agent-mode").unwrap(), "vibe");
        assert!(
            headers
                .get(AUTHORIZATION)
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("Bearer ")
        );
        assert_eq!(headers.get(CONNECTION).unwrap(), "close");
    }

    #[test]
    fn test_is_monthly_request_limit_detects_reason() {
        let body = r#"{"message":"You have reached the limit.","reason":"MONTHLY_REQUEST_COUNT"}"#;
        assert!(KiroProvider::is_monthly_request_limit(body));
    }

    #[test]
    fn test_is_monthly_request_limit_nested_reason() {
        let body = r#"{"error":{"reason":"MONTHLY_REQUEST_COUNT"}}"#;
        assert!(KiroProvider::is_monthly_request_limit(body));
    }

    #[test]
    fn test_is_monthly_request_limit_false() {
        let body = r#"{"message":"nope","reason":"DAILY_REQUEST_COUNT"}"#;
        assert!(!KiroProvider::is_monthly_request_limit(body));
    }

    #[test]
    fn test_is_invalid_bearer_token_true() {
        let body =
            r#"{"message":"The bearer token included in the request is invalid.","reason":null}"#;
        assert!(KiroProvider::is_invalid_bearer_token(body));
    }

    #[test]
    fn test_is_invalid_bearer_token_false() {
        let body = r#"{"message":"Forbidden","reason":null}"#;
        assert!(!KiroProvider::is_invalid_bearer_token(body));
    }

    #[test]
    #[cfg(not(feature = "sensitive-logs"))]
    fn test_summarize_error_body_extracts_message_and_reason() {
        let body =
            r#"{"message":"Input is too long.","reason":"CONTENT_LENGTH_EXCEEDS_THRESHOLD"}"#;
        let summary = KiroProvider::summarize_error_body(body);
        assert!(summary.contains("Input is too long"));
        assert!(summary.contains("CONTENT_LENGTH_EXCEEDS_THRESHOLD"));
    }

    #[test]
    #[cfg(not(feature = "sensitive-logs"))]
    fn test_summarize_error_body_extracts_nested_message_and_reason() {
        let body = r#"{"error":{"message":"Improperly formed request","reason":"BAD_REQUEST"}}"#;
        let summary = KiroProvider::summarize_error_body(body);
        assert!(summary.contains("Improperly formed request"));
        assert!(summary.contains("BAD_REQUEST"));
    }

    #[test]
    #[cfg(not(feature = "sensitive-logs"))]
    fn test_summarize_error_body_truncates_long_text() {
        let body = "x".repeat(1000);
        let summary = KiroProvider::summarize_error_body(&body);
        assert!(summary.len() <= 256 + 3);
        assert!(summary.ends_with("..."));
    }

    #[test]
    fn test_inject_profile_arn_with_social_auth() {
        // Social 认证且凭据有 profile_arn 时，应覆盖请求体中的 profileArn
        let mut credentials = KiroCredentials::default();
        credentials.auth_method = Some("social".to_string());
        credentials.profile_arn = Some("arn:aws:sso::111111111:profile/social-profile".to_string());

        let request_body =
            r#"{"conversationState":{},"profileArn":"arn:aws:sso::999999999:profile/old"}"#;
        let result = KiroProvider::inject_profile_arn(request_body, &credentials).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            parsed["profileArn"].as_str().unwrap(),
            "arn:aws:sso::111111111:profile/social-profile"
        );
    }

    #[test]
    fn test_inject_profile_arn_idc_removes_field() {
        // IDC 认证时，应移除请求体中的 profileArn 字段
        let mut credentials = KiroCredentials::default();
        credentials.auth_method = Some("idc".to_string());
        credentials.profile_arn = Some("arn:aws:sso::111111111:profile/idc-profile".to_string());

        let request_body =
            r#"{"conversationState":{},"profileArn":"arn:aws:sso::999999999:profile/old"}"#;
        let result = KiroProvider::inject_profile_arn(request_body, &credentials).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // profileArn 应被移除
        assert!(parsed.get("profileArn").is_none());
        // 其他字段应保留
        assert!(parsed.get("conversationState").is_some());
    }

    #[test]
    fn test_inject_profile_arn_builder_id_removes_field() {
        // Builder ID 认证时，应移除请求体中的 profileArn 字段
        let mut credentials = KiroCredentials::default();
        credentials.auth_method = Some("builder-id".to_string());

        let request_body =
            r#"{"conversationState":{},"profileArn":"arn:aws:sso::999999999:profile/old"}"#;
        let result = KiroProvider::inject_profile_arn(request_body, &credentials).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // profileArn 应被移除
        assert!(parsed.get("profileArn").is_none());
    }

    #[test]
    fn test_inject_profile_arn_aws_sso_oidc_by_client_credentials() {
        // 有 client_id + client_secret 时（AWS SSO OIDC 特征），应移除 profileArn
        let mut credentials = KiroCredentials::default();
        credentials.client_id = Some("client123".to_string());
        credentials.client_secret = Some("secret456".to_string());
        credentials.profile_arn = Some("arn:aws:sso::111111111:profile/test".to_string());

        let request_body =
            r#"{"conversationState":{},"profileArn":"arn:aws:sso::999999999:profile/old"}"#;
        let result = KiroProvider::inject_profile_arn(request_body, &credentials).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // profileArn 应被移除
        assert!(parsed.get("profileArn").is_none());
    }

    #[test]
    fn test_inject_profile_arn_without_credential_arn() {
        // Social 认证但凭据没有 profile_arn 时，应保持请求体不变
        let mut credentials = KiroCredentials::default();
        credentials.auth_method = Some("social".to_string());
        assert!(credentials.profile_arn.is_none());

        let request_body =
            r#"{"conversationState":{},"profileArn":"arn:aws:sso::999999999:profile/original"}"#;
        let result = KiroProvider::inject_profile_arn(request_body, &credentials).unwrap();

        // 应返回原始请求体（未修改）
        assert_eq!(result, request_body);
    }

    #[test]
    fn test_inject_profile_arn_adds_missing_field() {
        // Social 认证且请求体没有 profileArn 字段时，应添加
        let mut credentials = KiroCredentials::default();
        credentials.auth_method = Some("social".to_string());
        credentials.profile_arn = Some("arn:aws:sso::222222222:profile/new".to_string());

        let request_body = r#"{"conversationState":{"conversationId":"test"}}"#;
        let result = KiroProvider::inject_profile_arn(request_body, &credentials).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            parsed["profileArn"].as_str().unwrap(),
            "arn:aws:sso::222222222:profile/new"
        );
        // 确保原有字段保留
        assert_eq!(
            parsed["conversationState"]["conversationId"]
                .as_str()
                .unwrap(),
            "test"
        );
    }
}
