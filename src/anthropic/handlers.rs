//! Anthropic API Handler 函数

use std::convert::Infallible;

use crate::kiro::model::events::Event;
use crate::kiro::model::requests::kiro::KiroRequest;
use crate::kiro::parser::decoder::EventStreamDecoder;
use crate::token;
use anyhow::Error;
use axum::{
    Json as JsonExtractor,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};
use serde_json::json;
use std::time::Duration;
use tokio::time::{Instant, interval_at};
use uuid::Uuid;

/// 自适应压缩：最大迭代次数（避免极端输入导致过长 CPU 消耗）
const ADAPTIVE_COMPRESSION_MAX_ITERS: usize = 32;
/// tool_result 二次压缩的最低阈值（字符数）
const ADAPTIVE_MIN_TOOL_RESULT_MAX_CHARS: usize = 512;
/// tool_use input 二次压缩的最低阈值（字符数）
const ADAPTIVE_MIN_TOOL_USE_INPUT_MAX_CHARS: usize = 256;
/// 历史截断默认保留消息数（与 compressor.rs 的 preserve_count 保持一致）
const ADAPTIVE_HISTORY_PRESERVE_MESSAGES: usize = 2;
/// 消息内容二次压缩的最低阈值（字符数）
const ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS: usize = 8192;

use super::converter::{ConversionError, convert_request};
use super::middleware::AppState;
use super::stream::{BufferedStreamContext, SseEvent, StreamContext};
use super::types::{
    CountTokensRequest, CountTokensResponse, ErrorResponse, MessagesRequest, Model, ModelsResponse,
    OutputConfig, Thinking,
};
use super::websearch;

fn is_input_too_long_error(err: &Error) -> bool {
    // provider.rs 在遇到上游返回的 input-too-long 场景时，会在错误中保留以下关键字：
    // - CONTENT_LENGTH_EXCEEDS_THRESHOLD
    // - Input is too long
    //
    // 这类错误是确定性的请求问题（缩短输入才可恢复），不应返回 5xx（会诱发客户端重试）。
    // 注意：不包含 "Improperly formed request"，该错误可能由空消息内容等格式问题引起
    let s = err.to_string();
    s.contains("CONTENT_LENGTH_EXCEEDS_THRESHOLD") || s.contains("Input is too long")
}

fn is_quota_exhausted_error(err: &Error) -> bool {
    let s = err.to_string();
    s.contains("所有凭据已用尽")
}

fn is_improperly_formed_request_error(err: &Error) -> bool {
    let s = err.to_string();
    s.contains("Improperly formed request")
}

#[derive(Debug, Default, Clone, Copy)]
struct AdaptiveCompressionOutcome {
    initial_bytes: usize,
    final_bytes: usize,
    iters: usize,
    additional_history_turns_removed: usize,
    final_tool_result_max_chars: usize,
    final_tool_use_input_max_chars: usize,
    final_message_content_max_chars: usize,
}

/// 计算 KiroRequest 中所有图片 base64 数据的总字节数。
///
/// 该统计用于归因请求体大小（图片 base64 往往占用大量 bytes）。
/// 注意：上游存在请求体大小硬限制（约 5MiB），因此图片也必须控制体积；
/// `max_request_body_bytes` 的校验以实际序列化后的总字节数为准。
fn total_image_bytes(kiro_request: &KiroRequest) -> usize {
    let state = &kiro_request.conversation_state;
    let mut total = 0usize;

    // currentMessage 中的图片
    for img in &state.current_message.user_input_message.images {
        total += img.source.bytes.len();
    }

    // 历史消息中的图片
    for msg in &state.history {
        if let crate::kiro::model::requests::conversation::Message::User(user_msg) = msg {
            for img in &user_msg.user_input_message.images {
                total += img.source.bytes.len();
            }
        }
    }

    total
}

fn adaptive_shrink_request_body(
    kiro_request: &mut KiroRequest,
    base_config: &crate::model::config::CompressionConfig,
    max_body: usize,
    request_body: &mut String,
) -> Result<Option<AdaptiveCompressionOutcome>, serde_json::Error> {
    if max_body == 0 || request_body.len() <= max_body || !base_config.enabled {
        return Ok(None);
    }

    let mut outcome = AdaptiveCompressionOutcome {
        initial_bytes: request_body.len(),
        final_bytes: request_body.len(),
        iters: 0,
        additional_history_turns_removed: 0,
        final_tool_result_max_chars: base_config.tool_result_max_chars,
        final_tool_use_input_max_chars: base_config.tool_use_input_max_chars,
        final_message_content_max_chars: 0,
    };

    // 二次压缩策略：
    // 1) 逐步降低 tool_result_max_chars（仅当存在 tool_result/tools）
    // 2) 逐步降低 tool_use_input_max_chars（仅当存在 tool_use）
    // 3) 截断超长用户消息内容（当单条消息已超过阈值时优先）
    // 4) 按 request_body_bytes 成对移除最老的 user+assistant 两条消息（保留前 2 条）
    //
    // 每轮都会重新跑一次压缩管道（包含 tool 配对修复），再重新序列化计算字节数。
    let mut adaptive_config = base_config.clone();

    // 是否存在任何 tool_result / tools（否则降低阈值只会浪费迭代次数）
    let has_any_tool_results_or_tools = {
        let state = &kiro_request.conversation_state;
        if !state
            .current_message
            .user_input_message
            .user_input_message_context
            .tool_results
            .is_empty()
            || !state
                .current_message
                .user_input_message
                .user_input_message_context
                .tools
                .is_empty()
        {
            true
        } else {
            state.history.iter().any(|msg| match msg {
                crate::kiro::model::requests::conversation::Message::User(u) => {
                    !u.user_input_message
                        .user_input_message_context
                        .tool_results
                        .is_empty()
                        || !u
                            .user_input_message
                            .user_input_message_context
                            .tools
                            .is_empty()
                }
                _ => false,
            })
        }
    };

    // 是否存在任何 tool_use（否则降低阈值只会浪费迭代次数）
    let has_any_tool_uses = kiro_request
        .conversation_state
        .history
        .iter()
        .any(|msg| match msg {
            crate::kiro::model::requests::conversation::Message::Assistant(a) => a
                .assistant_response_message
                .tool_uses
                .as_ref()
                .is_some_and(|t| !t.is_empty()),
            _ => false,
        });

    // 扫描所有用户消息，找到最大 content 字符数作为初始 message_content_max_chars
    let max_content_chars = {
        let mut max_chars = kiro_request
            .conversation_state
            .current_message
            .user_input_message
            .content
            .chars()
            .count();
        for msg in &kiro_request.conversation_state.history {
            if let crate::kiro::model::requests::conversation::Message::User(u) = msg {
                max_chars = max_chars.max(u.user_input_message.content.chars().count());
            }
        }
        max_chars
    };
    // 初始值设为最大消息字符数的 3/4
    let mut message_content_max_chars =
        (max_content_chars * 3 / 4).max(ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS);

    for _ in 0..ADAPTIVE_COMPRESSION_MAX_ITERS {
        if request_body.len() <= max_body {
            break;
        }

        let mut changed = false;

        if has_any_tool_results_or_tools
            && adaptive_config.tool_result_max_chars > ADAPTIVE_MIN_TOOL_RESULT_MAX_CHARS
        {
            let next = (adaptive_config.tool_result_max_chars * 3 / 4)
                .max(ADAPTIVE_MIN_TOOL_RESULT_MAX_CHARS);
            if next < adaptive_config.tool_result_max_chars {
                adaptive_config.tool_result_max_chars = next;
                changed = true;
            }
        } else if has_any_tool_uses
            && adaptive_config.tool_use_input_max_chars > ADAPTIVE_MIN_TOOL_USE_INPUT_MAX_CHARS
        {
            let next = (adaptive_config.tool_use_input_max_chars * 3 / 4)
                .max(ADAPTIVE_MIN_TOOL_USE_INPUT_MAX_CHARS);
            if next < adaptive_config.tool_use_input_max_chars {
                adaptive_config.tool_use_input_max_chars = next;
                changed = true;
            }
        } else {
            // 如果任意单条 user content 已经超过 max_body，则移除历史并不能让请求落到阈值内，
            // 必须优先截断超长消息内容。
            let max_single_user_content_bytes = {
                let state = &kiro_request.conversation_state;
                let mut max_bytes = state.current_message.user_input_message.content.len();
                for msg in &state.history {
                    if let crate::kiro::model::requests::conversation::Message::User(u) = msg {
                        max_bytes = max_bytes.max(u.user_input_message.content.len());
                    }
                }
                max_bytes
            };

            let history = &mut kiro_request.conversation_state.history;
            if (max_single_user_content_bytes > max_body
                || history.len() <= ADAPTIVE_HISTORY_PRESERVE_MESSAGES + 2)
                && message_content_max_chars >= ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS
            {
                // 第三层：截断超长消息内容
                let saved = super::compressor::compress_long_messages_pass(
                    &mut kiro_request.conversation_state,
                    message_content_max_chars,
                );
                if saved > 0 {
                    changed = true;
                }
                // 记录本轮实际生效的阈值（递减前）
                outcome.final_message_content_max_chars = message_content_max_chars;
                // 每轮递减 3/4
                message_content_max_chars =
                    (message_content_max_chars * 3 / 4).max(ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS);
            } else if history.len() > ADAPTIVE_HISTORY_PRESERVE_MESSAGES + 2 {
                // 第四层：移除最老历史消息（成对移除 user+assistant）
                let preserve = ADAPTIVE_HISTORY_PRESERVE_MESSAGES;
                let min_len = preserve + 2;
                let removable = history.len().saturating_sub(min_len);
                // 单轮最多移除 16 条消息（8 轮），避免一次性丢弃过多上下文
                let mut remove_msgs = removable.min(16);
                remove_msgs -= remove_msgs % 2; // 保持成对移除
                if remove_msgs > 0 {
                    history.drain(preserve..preserve + remove_msgs);
                    outcome.additional_history_turns_removed += remove_msgs / 2;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }

        super::compressor::compress(&mut kiro_request.conversation_state, &adaptive_config);
        *request_body = serde_json::to_string(kiro_request)?;
        outcome.iters += 1;
        outcome.final_bytes = request_body.len();
    }

    outcome.final_tool_result_max_chars = adaptive_config.tool_result_max_chars;
    outcome.final_tool_use_input_max_chars = adaptive_config.tool_use_input_max_chars;
    // final_message_content_max_chars 在循环内截断时已记录实际生效值；
    // 若第四层从未执行，保持默认 0 表示未触发

    Ok(Some(outcome))
}

fn map_kiro_provider_error_to_response(request_body: &str, err: Error) -> Response {
    if is_input_too_long_error(&err) {
        tracing::warn!(
            kiro_request_body_bytes = request_body.len(),
            error = %err,
            "上游拒绝请求：输入上下文过长（不应重试）"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request_error",
                "Input is too long (CONTENT_LENGTH_EXCEEDS_THRESHOLD). Reduce conversation history/system/tools; retrying the same request will not help.",
            )),
        )
            .into_response();
    }

    if is_improperly_formed_request_error(&err) {
        tracing::warn!(
            error = %err,
            "上游拒绝请求：请求格式错误（可能是空消息内容或其他格式问题）"
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request_error",
                "Improperly formed request. Check message content is not empty and request format is valid.",
            )),
        )
            .into_response();
    }

    if is_quota_exhausted_error(&err) {
        tracing::warn!(error = %err, "所有凭据配额已耗尽");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse::new(
                "rate_limit_error",
                "All credentials quota exhausted. Please wait for quota reset or add new credentials.",
            )),
        )
            .into_response();
    }

    tracing::error!("Kiro API 调用失败: {}", err);
    #[cfg(feature = "sensitive-logs")]
    tracing::error!(
        "上游报错，完整请求体（用于诊断）: {}",
        truncate_base64_in_request_body(request_body)
    );
    (
        StatusCode::BAD_GATEWAY,
        Json(ErrorResponse::new(
            "api_error",
            format!("上游 API 调用失败: {}", err),
        )),
    )
        .into_response()
}

/// 对 user_id 进行掩码处理，保护隐私
fn mask_user_id(user_id: Option<&str>) -> String {
    match user_id {
        Some(id) => {
            let chars: Vec<char> = id.chars().collect();
            let len = chars.len();
            if len > 25 {
                format!(
                    "{}***{}",
                    chars[..13].iter().collect::<String>(),
                    chars[len - 8..].iter().collect::<String>()
                )
            } else if len > 12 {
                format!(
                    "{}***{}",
                    chars[..4].iter().collect::<String>(),
                    chars[len - 4..].iter().collect::<String>()
                )
            } else {
                "***".to_string()
            }
        }
        None => "None".to_string(),
    }
}

/// GET /v1/models
///
/// 返回可用的模型列表
pub async fn get_models() -> impl IntoResponse {
    tracing::info!("Received GET /v1/models request");

    let models = vec![
        Model {
            id: "claude-sonnet-4-6".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.6".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(1_000_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-sonnet-4-6-thinking".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.6 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(1_000_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-sonnet-4-6-agentic".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.6 (Agentic)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(1_000_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-sonnet-4-5-20250929".to_string(),
            object: "model".to_string(),
            created: 1727568000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-sonnet-4-5-20250929-thinking".to_string(),
            object: "model".to_string(),
            created: 1727568000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-sonnet-4-5-20250929-agentic".to_string(),
            object: "model".to_string(),
            created: 1727568000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.5 (Agentic)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-opus-4-5-20251101".to_string(),
            object: "model".to_string(),
            created: 1730419200,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-opus-4-5-20251101-thinking".to_string(),
            object: "model".to_string(),
            created: 1730419200,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-opus-4-5-20251101-agentic".to_string(),
            object: "model".to_string(),
            created: 1730419200,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.5 (Agentic)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-opus-4-6".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.6".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(1_000_000),
            max_completion_tokens: Some(128_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-opus-4-6-thinking".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.6 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(1_000_000),
            max_completion_tokens: Some(128_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-opus-4-6-agentic".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.6 (Agentic)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(1_000_000),
            max_completion_tokens: Some(128_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-haiku-4-5-20251001".to_string(),
            object: "model".to_string(),
            created: 1727740800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Haiku 4.5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-haiku-4-5-20251001-thinking".to_string(),
            object: "model".to_string(),
            created: 1727740800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Haiku 4.5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
        Model {
            id: "claude-haiku-4-5-20251001-agentic".to_string(),
            object: "model".to_string(),
            created: 1727740800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Haiku 4.5 (Agentic)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
            context_length: Some(200_000),
            max_completion_tokens: Some(64_000),
            thinking: Some(true),
        },
    ];

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

/// POST /v1/messages
///
/// 创建消息（对话）
pub async fn post_messages(
    State(state): State<AppState>,
    JsonExtractor(mut payload): JsonExtractor<MessagesRequest>,
) -> Response {
    // 检测模型名是否包含 "thinking" 后缀，若包含则覆写 thinking 配置
    override_thinking_from_model_name(&mut payload);

    // 提取 user_id 用于凭据亲和性
    let user_id = payload.metadata.as_ref().and_then(|m| m.user_id.clone());

    // 估算压缩前 input tokens（需在 convert_request 之前，因为后者会消费压缩）
    let estimated_input_tokens = token::count_all_tokens(
        payload.model.clone(),
        payload.system.clone(),
        payload.messages.clone(),
        payload.tools.clone(),
    ) as i32;

    tracing::info!(
        model = %payload.model,
        max_tokens = %payload.max_tokens,
        stream = %payload.stream,
        message_count = %payload.messages.len(),
        user_id = %mask_user_id(user_id.as_deref()),
        estimated_input_tokens,
        "Received POST /v1/messages request"
    );
    // 检查 KiroProvider 是否可用
    let provider = match &state.kiro_provider {
        Some(p) => p.clone(),
        None => {
            tracing::error!("KiroProvider 未配置");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(
                    "service_unavailable",
                    "Kiro API provider not configured",
                )),
            )
                .into_response();
        }
    };

    // 检查是否为纯 WebSearch 请求（仅 web_search 单工具 / tool_choice 强制 / 前缀匹配）
    if websearch::should_handle_websearch_request(&payload) {
        tracing::info!("检测到纯 WebSearch 请求，路由到本地 WebSearch 处理");
        return websearch::handle_websearch_request(provider, &payload, estimated_input_tokens)
            .await;
    }

    // 混合工具场景：剔除 web_search 后转发上游
    if websearch::has_web_search_tool(&payload) {
        tracing::info!("检测到混合工具列表中的 web_search，剔除后转发上游");
        websearch::strip_web_search_tools(&mut payload);
    }

    // 转换请求
    let conversion_result = match convert_request(&payload, &state.compression_config) {
        Ok(result) => result,
        Err(e) => {
            let (error_type, message) = match &e {
                ConversionError::UnsupportedModel(model) => {
                    ("invalid_request_error", format!("模型不支持: {}", model))
                }
                ConversionError::EmptyMessages => {
                    ("invalid_request_error", "消息列表为空".to_string())
                }
                ConversionError::EmptyMessageContent => {
                    ("invalid_request_error", "消息内容为空".to_string())
                }
            };
            tracing::warn!("请求转换失败: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(error_type, message)),
            )
                .into_response();
        }
    };

    // 输出压缩统计（以字节为单位；用于排查上游请求体大小限制，实测约 5MiB 左右会触发 400）
    if let Some(ref stats) = conversion_result.compression_stats {
        tracing::info!(
            estimated_input_tokens,
            bytes_saved_total = stats.total_saved(),
            whitespace_bytes_saved = stats.whitespace_saved,
            thinking_bytes_saved = stats.thinking_saved,
            tool_result_bytes_saved = stats.tool_result_saved,
            tool_use_input_bytes_saved = stats.tool_use_input_saved,
            history_turns_removed = stats.history_turns_removed,
            history_bytes_saved = stats.history_bytes_saved,
            "输入压缩完成"
        );
    }

    // 构建 Kiro 请求
    let mut kiro_request = KiroRequest {
        conversation_state: conversion_result.conversation_state,
        profile_arn: state.profile_arn.clone(),
    };

    let mut request_body = match serde_json::to_string(&kiro_request) {
        Ok(body) => body,
        Err(e) => {
            tracing::error!("序列化请求失败: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    "internal_error",
                    format!("序列化请求失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 请求体大小预检（上游存在硬性请求体大小限制；按实际序列化后的总字节数判断）
    let max_body = state.compression_config.max_request_body_bytes;
    if max_body > 0 && request_body.len() > max_body && state.compression_config.enabled {
        // 自适应二次压缩：按 request_body_bytes 迭代截断，尽量把请求缩到阈值内
        match adaptive_shrink_request_body(
            &mut kiro_request,
            &state.compression_config,
            max_body,
            &mut request_body,
        ) {
            Ok(Some(outcome)) => {
                tracing::warn!(
                    conversation_id = kiro_request.conversation_state.conversation_id.as_str(),
                    initial_bytes = outcome.initial_bytes,
                    final_bytes = outcome.final_bytes,
                    threshold = max_body,
                    iters = outcome.iters,
                    additional_history_turns_removed = outcome.additional_history_turns_removed,
                    final_tool_result_max_chars = outcome.final_tool_result_max_chars,
                    final_tool_use_input_max_chars = outcome.final_tool_use_input_max_chars,
                    final_message_content_max_chars = outcome.final_message_content_max_chars,
                    "请求体超过阈值，已执行自适应二次压缩"
                );
            }
            Ok(None) => {}
            Err(e) => {
                tracing::error!("自适应二次压缩序列化失败: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new(
                        "internal_error",
                        format!("序列化请求失败: {}", e),
                    )),
                )
                    .into_response();
            }
        }
    }

    // 压缩后再次检查（输出 image_bytes/non-image bytes 便于排查）
    let final_img_bytes = total_image_bytes(&kiro_request);
    let final_effective_len = request_body.len().saturating_sub(final_img_bytes);
    if max_body > 0 && request_body.len() > max_body {
        tracing::warn!(
            conversation_id = kiro_request.conversation_state.conversation_id.as_str(),
            request_body_bytes = request_body.len(),
            image_bytes = final_img_bytes,
            effective_bytes = final_effective_len,
            threshold = max_body,
            "请求体超过安全阈值，拒绝发送"
        );
        #[cfg(feature = "sensitive-logs")]
        tracing::error!(
            "自适应压缩仍超限，完整请求体（用于诊断）: {}",
            truncate_base64_in_request_body(&request_body)
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request_error",
                format!(
                    "Request too large ({} bytes total; images {} bytes; non-image {} bytes; limit {}). Reduce conversation history/tool output or number/size of images.",
                    request_body.len(),
                    final_img_bytes,
                    final_effective_len,
                    max_body
                ),
            )),
        )
            .into_response();
    }

    tracing::debug!(
        kiro_request_body_bytes = request_body.len(),
        "已构建 Kiro 请求体"
    );

    // 检查是否启用了thinking
    let thinking_enabled = payload
        .thinking
        .as_ref()
        .map(|t| t.is_enabled())
        .unwrap_or(false);

    if payload.stream {
        // 流式响应
        handle_stream_request(
            provider,
            &request_body,
            &payload.model,
            estimated_input_tokens,
            thinking_enabled,
            user_id.as_deref(),
        )
        .await
    } else {
        // 非流式响应
        handle_non_stream_request(
            provider,
            &request_body,
            &payload.model,
            estimated_input_tokens,
            user_id.as_deref(),
        )
        .await
    }
}
async fn handle_stream_request(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
    thinking_enabled: bool,
    user_id: Option<&str>,
) -> Response {
    // 调用 Kiro API（支持多凭据故障转移）
    let response = match provider.call_api_stream(request_body, user_id).await {
        Ok(resp) => resp,
        Err(e) => return map_kiro_provider_error_to_response(request_body, e),
    };

    // 创建流处理上下文
    let mut ctx = StreamContext::new_with_thinking(model, input_tokens, thinking_enabled);

    // 生成初始事件
    let initial_events = ctx.generate_initial_events();

    // 创建 SSE 流
    let stream = create_sse_stream(response, ctx, initial_events);

    // 返回 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Ping 事件间隔（25秒）
const PING_INTERVAL_SECS: u64 = 25;

/// 创建 ping 事件的 SSE 字符串
fn create_ping_sse() -> Bytes {
    Bytes::from("event: ping\ndata: {\"type\": \"ping\"}\n\n")
}

/// 创建 SSE 事件流
fn create_sse_stream(
    response: reqwest::Response,
    ctx: StreamContext,
    initial_events: Vec<SseEvent>,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    // 先发送初始事件
    let initial_stream = stream::iter(
        initial_events
            .into_iter()
            .map(|e| Ok(Bytes::from(e.to_sse_string()))),
    );

    // 然后处理 Kiro 响应流，同时每25秒发送 ping 保活
    let body_stream = response.bytes_stream();
    let ping_period = Duration::from_secs(PING_INTERVAL_SECS);
    let ping_interval = interval_at(Instant::now() + ping_period, ping_period);

    let processing_stream = stream::unfold(
        (body_stream, ctx, EventStreamDecoder::new(), false, ping_interval),
        |(mut body_stream, mut ctx, mut decoder, finished, mut ping_interval)| async move {
            if finished {
                return None;
            }

            // 使用 select! 同时等待数据和 ping 定时器
            tokio::select! {
                // 处理数据流
                chunk_result = body_stream.next() => {
                    match chunk_result {
                        Some(Ok(chunk)) => {
                            // 解码事件
                            if let Err(e) = decoder.feed(&chunk) {
                                tracing::warn!("缓冲区溢出: {}", e);
                            }

                            let mut events = Vec::new();
                            for result in decoder.decode_iter() {
                                match result {
                                    Ok(frame) => {
                                        if let Ok(event) = Event::from_frame(frame) {
                                            let sse_events = ctx.process_kiro_event(&event);
                                            events.extend(sse_events);
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!("解码事件失败: {}", e);
                                    }
                                }
                            }

                            // 转换为 SSE 字节流
                            let bytes: Vec<Result<Bytes, Infallible>> = events
                                .into_iter()
                                .map(|e| Ok(Bytes::from(e.to_sse_string())))
                                .collect();

                            Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval)))
                        }
                        Some(Err(e)) => {
                            tracing::error!("读取响应流失败: {}", e);
                            // 发送最终事件并结束
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|e| Ok(Bytes::from(e.to_sse_string())))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval)))
                        }
                        None => {
                            // 流结束，发送最终事件
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|e| Ok(Bytes::from(e.to_sse_string())))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval)))
                        }
                    }
                }
                // 发送 ping 保活
                _ = ping_interval.tick() => {
                    tracing::trace!("发送 ping 保活事件");
                    let bytes: Vec<Result<Bytes, Infallible>> = vec![Ok(create_ping_sse())];
                    Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval)))
                }
            }
        },
    )
    .flatten();

    initial_stream.chain(processing_stream)
}

/// 处理非流式请求
async fn handle_non_stream_request(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
    user_id: Option<&str>,
) -> Response {
    // 调用 Kiro API（支持多凭据故障转移）
    let response = match provider.call_api(request_body, user_id).await {
        Ok(resp) => resp,
        Err(e) => return map_kiro_provider_error_to_response(request_body, e),
    };

    // 读取响应体
    let body_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("读取响应体失败: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::new(
                    "api_error",
                    format!("读取响应失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 解析事件流
    let mut decoder = EventStreamDecoder::new();
    if let Err(e) = decoder.feed(&body_bytes) {
        tracing::warn!("缓冲区溢出: {}", e);
    }

    let mut text_content = String::new();
    let mut tool_uses: Vec<serde_json::Value> = Vec::new();
    let mut has_tool_use = false;
    let mut stop_reason = "end_turn".to_string();
    // 从 contextUsageEvent 计算的实际输入 tokens
    let mut context_input_tokens: Option<i32> = None;

    // 收集工具调用的增量 JSON
    let mut tool_json_buffers: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for result in decoder.decode_iter() {
        match result {
            Ok(frame) => {
                if let Ok(event) = Event::from_frame(frame) {
                    match event {
                        Event::AssistantResponse(resp) => {
                            text_content.push_str(&resp.content);
                        }
                        Event::ToolUse(tool_use) => {
                            has_tool_use = true;

                            // 累积工具的 JSON 输入
                            let buffer = tool_json_buffers
                                .entry(tool_use.tool_use_id.clone())
                                .or_default();
                            buffer.push_str(&tool_use.input);

                            // 如果是完整的工具调用，添加到列表
                            if tool_use.stop {
                                let input: serde_json::Value = if buffer.trim().is_empty() {
                                    // 上游可能省略无参工具的 input 字段（或传空字符串）。
                                    // 这里将其视为合法的空对象，避免 EOF 解析错误导致日志噪音。
                                    serde_json::json!({})
                                } else {
                                    serde_json::from_str(buffer).unwrap_or_else(|e| {
                                        // 检测是否为截断导致的解析失败
                                        if let Some(truncation_info) =
                                            super::truncation::detect_truncation(
                                                &tool_use.name,
                                                &tool_use.tool_use_id,
                                                buffer,
                                            )
                                        {
                                            let soft_msg =
                                                super::truncation::build_soft_failure_result(
                                                    &truncation_info,
                                                );
                                            tracing::warn!(
                                                tool_use_id = %tool_use.tool_use_id,
                                                truncation_type = %truncation_info.truncation_type,
                                                "检测到工具调用截断: {}", soft_msg
                                            );
                                        }

                                        // 仅在显式开启敏感日志时输出完整内容
                                        #[cfg(feature = "sensitive-logs")]
                                        tracing::warn!(
                                            tool_use_id = %tool_use.tool_use_id,
                                            buffer = %buffer,
                                            request_body = %truncate_middle(request_body, 1200),
                                            "工具输入 JSON 解析失败: {e}"
                                        );
                                        #[cfg(not(feature = "sensitive-logs"))]
                                        tracing::warn!(
                                            tool_use_id = %tool_use.tool_use_id,
                                            buffer_bytes = buffer.len(),
                                            request_body_bytes = request_body.len(),
                                            "工具输入 JSON 解析失败: {e}"
                                        );
                                        serde_json::json!({})
                                    })
                                };

                                // 释放已完成的 buffer，避免请求处理期间内存重复占用
                                tool_json_buffers.remove(&tool_use.tool_use_id);

                                tool_uses.push(json!({
                                    "type": "tool_use",
                                    "id": tool_use.tool_use_id,
                                    "name": tool_use.name,
                                    "input": input
                                }));
                            }
                        }
                        Event::ContextUsage(context_usage) => {
                            // 从上下文使用百分比计算实际的 input_tokens
                            let context_window =
                                super::types::get_context_window_size(model) as f64;
                            let actual_input_tokens =
                                (context_usage.context_usage_percentage * context_window / 100.0)
                                    as i32;
                            context_input_tokens = Some(actual_input_tokens);
                            // 上下文使用量达到 100% 时，设置 stop_reason 为 model_context_window_exceeded
                            if context_usage.context_usage_percentage >= 100.0 {
                                stop_reason = "model_context_window_exceeded".to_string();
                            }
                            tracing::debug!(
                                "收到 contextUsageEvent: {}%, 计算 input_tokens: {} (context_window: {})",
                                context_usage.context_usage_percentage,
                                actual_input_tokens,
                                context_window as i32
                            );
                        }
                        Event::Exception { exception_type, .. } => {
                            if exception_type == "ContentLengthExceededException" {
                                stop_reason = "max_tokens".to_string();
                            }
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                tracing::warn!("解码事件失败: {}", e);
            }
        }
    }

    // 确定 stop_reason
    if has_tool_use && stop_reason == "end_turn" {
        stop_reason = "tool_use".to_string();
    }

    // 构建响应内容
    let mut content: Vec<serde_json::Value> = Vec::new();

    if !text_content.is_empty() {
        content.push(json!({
            "type": "text",
            "text": text_content
        }));
    }

    content.extend(tool_uses);

    // 估算输出 tokens
    let output_tokens = token::estimate_output_tokens(&content);

    // 使用从 contextUsageEvent 计算的 input_tokens，如果没有则使用估算值
    let final_input_tokens = context_input_tokens.unwrap_or(input_tokens);

    // 构建 Anthropic 响应
    let response_body = json!({
        "id": format!("msg_{}", Uuid::new_v4().to_string().replace('-', "")),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": final_input_tokens,
            "output_tokens": output_tokens
        }
    });

    (StatusCode::OK, Json(response_body)).into_response()
}

/// 检测模型名是否包含 "thinking" 后缀，若包含则覆写 thinking 配置
///
/// 支持的后缀格式：
/// - `-thinking-minimal` → budget 512
/// - `-thinking-low` → budget 1024
/// - `-thinking-medium` → budget 8192
/// - `-thinking-high` → budget 24576
/// - `-thinking-xhigh` → budget 32768
/// - `-thinking` → budget 20000（默认）
///
/// - Opus 4.6：覆写为 adaptive 类型
/// - 其他模型：覆写为 enabled 类型
fn override_thinking_from_model_name(payload: &mut MessagesRequest) {
    let model_lower = payload.model.to_lowercase();
    if !model_lower.contains("thinking") {
        return;
    }

    // 具体后缀必须在通用 "thinking" 之前匹配
    let budget_tokens = if model_lower.ends_with("-thinking-minimal") {
        512
    } else if model_lower.ends_with("-thinking-low") {
        1024
    } else if model_lower.ends_with("-thinking-medium") {
        8192
    } else if model_lower.ends_with("-thinking-high") {
        24576
    } else if model_lower.ends_with("-thinking-xhigh") {
        32768
    } else if model_lower.ends_with("-thinking") {
        20000
    } else {
        // "thinking" 出现在模型名中但不是后缀（如 "thinking-model-v2"），不覆写
        return;
    };

    let is_opus_or_sonnet_4_6 = (model_lower.contains("opus") || model_lower.contains("sonnet"))
        && (model_lower.contains("4-6") || model_lower.contains("4.6"));

    let thinking_type = if is_opus_4_6 { "adaptive" } else { "enabled" };

    tracing::info!(
        model = %payload.model,
        thinking_type = thinking_type,
        budget_tokens = budget_tokens,
        "模型名包含 thinking 后缀，覆写 thinking 配置"
    );

    payload.thinking = Some(Thinking {
        thinking_type: thinking_type.to_string(),
        budget_tokens,
    });

    if is_opus_or_sonnet_4_6 {
        payload.output_config = Some(OutputConfig {
            effort: "high".to_string(),
        });
    }
}

/// POST /v1/messages/count_tokens
///
/// 计算消息的 token 数量
pub async fn count_tokens(
    JsonExtractor(payload): JsonExtractor<CountTokensRequest>,
) -> impl IntoResponse {
    tracing::info!(
        model = %payload.model,
        message_count = %payload.messages.len(),
        "Received POST /v1/messages/count_tokens request"
    );

    let total_tokens = token::count_all_tokens(
        payload.model.clone(),
        payload.system.clone(),
        payload.messages.clone(),
        payload.tools.clone(),
    ) as i32;

    Json(CountTokensResponse {
        input_tokens: total_tokens.max(1) as i32,
    })
}

/// POST /cc/v1/messages
///
/// Claude Code 兼容端点，与 /v1/messages 的区别在于：
/// - 流式响应会等待 kiro 端返回 contextUsageEvent 后再发送 message_start
/// - message_start 中的 input_tokens 是从 contextUsageEvent 计算的准确值
pub async fn post_messages_cc(
    State(state): State<AppState>,
    JsonExtractor(mut payload): JsonExtractor<MessagesRequest>,
) -> Response {
    // 检查 KiroProvider 是否可用
    let provider = match &state.kiro_provider {
        Some(p) => p.clone(),
        None => {
            tracing::error!("KiroProvider 未配置");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(
                    "service_unavailable",
                    "Kiro API provider not configured",
                )),
            )
                .into_response();
        }
    };

    // 检测模型名是否包含 "thinking" 后缀，若包含则覆写 thinking 配置
    override_thinking_from_model_name(&mut payload);

    // 估算压缩前 input tokens（需在 convert_request 之前）
    let estimated_input_tokens = token::count_all_tokens(
        payload.model.clone(),
        payload.system.clone(),
        payload.messages.clone(),
        payload.tools.clone(),
    ) as i32;

    tracing::info!(
        model = %payload.model,
        max_tokens = %payload.max_tokens,
        stream = %payload.stream,
        message_count = %payload.messages.len(),
        estimated_input_tokens,
        "Received POST /cc/v1/messages request"
    );

    // 检查是否为纯 WebSearch 请求（仅 web_search 单工具 / tool_choice 强制 / 前缀匹配）
    if websearch::should_handle_websearch_request(&payload) {
        tracing::info!("检测到纯 WebSearch 请求，路由到本地 WebSearch 处理");
        return websearch::handle_websearch_request(provider, &payload, estimated_input_tokens)
            .await;
    }

    // 混合工具场景：剔除 web_search 后转发上游
    if websearch::has_web_search_tool(&payload) {
        tracing::info!("检测到混合工具列表中的 web_search，剔除后转发上游");
        websearch::strip_web_search_tools(&mut payload);
    }

    // 转换请求
    let conversion_result = match convert_request(&payload, &state.compression_config) {
        Ok(result) => result,
        Err(e) => {
            let (error_type, message) = match &e {
                ConversionError::UnsupportedModel(model) => {
                    ("invalid_request_error", format!("模型不支持: {}", model))
                }
                ConversionError::EmptyMessages => {
                    ("invalid_request_error", "消息列表为空".to_string())
                }
                ConversionError::EmptyMessageContent => {
                    ("invalid_request_error", "消息内容为空".to_string())
                }
            };
            tracing::warn!("请求转换失败: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(error_type, message)),
            )
                .into_response();
        }
    };

    // 输出压缩统计（以字节为单位；用于排查上游请求体大小限制，实测约 5MiB 左右会触发 400）
    if let Some(ref stats) = conversion_result.compression_stats {
        tracing::info!(
            estimated_input_tokens,
            bytes_saved_total = stats.total_saved(),
            whitespace_bytes_saved = stats.whitespace_saved,
            thinking_bytes_saved = stats.thinking_saved,
            tool_result_bytes_saved = stats.tool_result_saved,
            tool_use_input_bytes_saved = stats.tool_use_input_saved,
            history_turns_removed = stats.history_turns_removed,
            history_bytes_saved = stats.history_bytes_saved,
            "输入压缩完成"
        );
    }

    // 构建 Kiro 请求
    let mut kiro_request = KiroRequest {
        conversation_state: conversion_result.conversation_state,
        profile_arn: state.profile_arn.clone(),
    };

    let mut request_body = match serde_json::to_string(&kiro_request) {
        Ok(body) => body,
        Err(e) => {
            tracing::error!("序列化请求失败: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    "internal_error",
                    format!("序列化请求失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 请求体大小预检（上游存在硬性请求体大小限制；按实际序列化后的总字节数判断）
    let max_body = state.compression_config.max_request_body_bytes;
    if max_body > 0 && request_body.len() > max_body && state.compression_config.enabled {
        // 自适应二次压缩：按 request_body_bytes 迭代截断，尽量把请求缩到阈值内
        match adaptive_shrink_request_body(
            &mut kiro_request,
            &state.compression_config,
            max_body,
            &mut request_body,
        ) {
            Ok(Some(outcome)) => {
                tracing::warn!(
                    conversation_id = kiro_request.conversation_state.conversation_id.as_str(),
                    initial_bytes = outcome.initial_bytes,
                    final_bytes = outcome.final_bytes,
                    threshold = max_body,
                    iters = outcome.iters,
                    additional_history_turns_removed = outcome.additional_history_turns_removed,
                    final_tool_result_max_chars = outcome.final_tool_result_max_chars,
                    final_tool_use_input_max_chars = outcome.final_tool_use_input_max_chars,
                    final_message_content_max_chars = outcome.final_message_content_max_chars,
                    "请求体超过阈值，已执行自适应二次压缩"
                );
            }
            Ok(None) => {}
            Err(e) => {
                tracing::error!("自适应二次压缩序列化失败: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new(
                        "internal_error",
                        format!("序列化请求失败: {}", e),
                    )),
                )
                    .into_response();
            }
        }
    }

    // 压缩后再次检查（输出 image_bytes/non-image bytes 便于排查）
    let final_img_bytes = total_image_bytes(&kiro_request);
    let final_effective_len = request_body.len().saturating_sub(final_img_bytes);
    if max_body > 0 && request_body.len() > max_body {
        tracing::warn!(
            conversation_id = kiro_request.conversation_state.conversation_id.as_str(),
            request_body_bytes = request_body.len(),
            image_bytes = final_img_bytes,
            effective_bytes = final_effective_len,
            threshold = max_body,
            "请求体超过安全阈值，拒绝发送"
        );
        #[cfg(feature = "sensitive-logs")]
        tracing::error!(
            "自适应压缩仍超限，完整请求体（用于诊断）: {}",
            truncate_base64_in_request_body(&request_body)
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request_error",
                format!(
                    "Request too large ({} bytes total; images {} bytes; non-image {} bytes; limit {}). Reduce conversation history/tool output or number/size of images.",
                    request_body.len(),
                    final_img_bytes,
                    final_effective_len,
                    max_body
                ),
            )),
        )
            .into_response();
    }

    tracing::debug!(
        kiro_request_body_bytes = request_body.len(),
        "已构建 Kiro 请求体"
    );

    // 检查是否启用了thinking
    let thinking_enabled = payload
        .thinking
        .as_ref()
        .map(|t| t.is_enabled())
        .unwrap_or(false);

    if payload.stream {
        // 流式响应（缓冲模式）
        let user_id = payload.metadata.as_ref().and_then(|m| m.user_id.as_deref());
        handle_stream_request_buffered(
            provider,
            &request_body,
            &payload.model,
            estimated_input_tokens,
            thinking_enabled,
            user_id,
        )
        .await
    } else {
        // 非流式响应（复用现有逻辑，已经使用正确的 input_tokens）
        let user_id = payload.metadata.as_ref().and_then(|m| m.user_id.as_deref());
        handle_non_stream_request(
            provider,
            &request_body,
            &payload.model,
            estimated_input_tokens,
            user_id,
        )
        .await
    }
}

/// 处理流式请求（缓冲版本）
///
/// 与 `handle_stream_request` 不同，此函数会缓冲所有事件直到流结束，
/// 然后用从 contextUsageEvent 计算的正确 input_tokens 生成 message_start 事件。
async fn handle_stream_request_buffered(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    model: &str,
    estimated_input_tokens: i32,
    thinking_enabled: bool,
    user_id: Option<&str>,
) -> Response {
    // 调用 Kiro API（支持多凭据故障转移）
    let response = match provider.call_api_stream(request_body, user_id).await {
        Ok(resp) => resp,
        Err(e) => return map_kiro_provider_error_to_response(request_body, e),
    };

    // 创建缓冲流处理上下文
    let ctx = BufferedStreamContext::new(model, estimated_input_tokens, thinking_enabled);

    // 创建缓冲 SSE 流
    let stream = create_buffered_sse_stream(response, ctx);

    // 返回 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// 创建缓冲 SSE 事件流
///
/// 工作流程：
/// 1. 等待上游流完成，期间只发送 ping 保活信号
/// 2. 使用 StreamContext 的事件处理逻辑处理所有 Kiro 事件，结果缓存
/// 3. 流结束后，用正确的 input_tokens 更正 message_start 事件
/// 4. 一次性发送所有事件
fn create_buffered_sse_stream(
    response: reqwest::Response,
    ctx: BufferedStreamContext,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let body_stream = response.bytes_stream();
    let ping_period = Duration::from_secs(PING_INTERVAL_SECS);
    let ping_interval = interval_at(Instant::now() + ping_period, ping_period);

    stream::unfold(
        (
            body_stream,
            ctx,
            EventStreamDecoder::new(),
            false,
            ping_interval,
        ),
        |(mut body_stream, mut ctx, mut decoder, finished, mut ping_interval)| async move {
            if finished {
                return None;
            }

            loop {
                tokio::select! {
                    // 使用 biased 模式，优先检查 ping 定时器
                    // 避免在上游 chunk 密集时 ping 被"饿死"
                    biased;

                    // 优先检查 ping 保活（等待期间唯一发送的数据）
                    _ = ping_interval.tick() => {
                        tracing::trace!("发送 ping 保活事件（缓冲模式）");
                        let bytes: Vec<Result<Bytes, Infallible>> = vec![Ok(create_ping_sse())];
                        return Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval)));
                    }

                    // 然后处理数据流
                    chunk_result = body_stream.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                // 解码事件
                                if let Err(e) = decoder.feed(&chunk) {
                                    tracing::warn!("缓冲区溢出: {}", e);
                                }

                                for result in decoder.decode_iter() {
                                    match result {
                                        Ok(frame) => {
                                            if let Ok(event) = Event::from_frame(frame) {
                                                // 缓冲事件（复用 StreamContext 的处理逻辑）
                                                ctx.process_and_buffer(&event);
                                            }
                                        }
                                        Err(e) => {
                                            tracing::warn!("解码事件失败: {}", e);
                                        }
                                    }
                                }
                                // 继续读取下一个 chunk，不发送任何数据
                            }
                            Some(Err(e)) => {
                                tracing::error!("读取响应流失败: {}", e);
                                // 发生错误，完成处理并返回所有事件
                                let all_events = ctx.finish_and_get_all_events();
                                let bytes: Vec<Result<Bytes, Infallible>> = all_events
                                    .into_iter()
                                    .map(|e| Ok(Bytes::from(e.to_sse_string())))
                                    .collect();
                                return Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval)));
                            }
                            None => {
                                // 流结束，完成处理并返回所有事件（已更正 input_tokens）
                                let all_events = ctx.finish_and_get_all_events();
                                let bytes: Vec<Result<Bytes, Infallible>> = all_events
                                    .into_iter()
                                    .map(|e| Ok(Bytes::from(e.to_sse_string())))
                                    .collect();
                                return Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval)));
                            }
                        }
                    }
                }
            }
        },
    )
    .flatten()
}

/// 截断字符串中间部分，保留头尾各 `keep` 个字符
///
/// 用于 debug 日志：避免输出过长的请求体，同时保留足够上下文便于排查。
/// 正确处理 UTF-8 多字节字符边界，不会截断中文。
#[cfg(feature = "sensitive-logs")]
fn truncate_middle(s: &str, keep: usize) -> std::borrow::Cow<'_, str> {
    // 按字符数计算，避免截断后反而更长
    let char_count = s.chars().count();
    let min_omit = 30; // 省略号 + 数字的最小开销，确保截断有意义
    if char_count <= keep * 2 + min_omit {
        return std::borrow::Cow::Borrowed(s);
    }

    // 找到第 keep 个字符的字节边界
    let head_end = s
        .char_indices()
        .nth(keep)
        .map(|(i, _)| i)
        .unwrap_or(s.len());

    // 找到倒数第 keep 个字符的字节边界
    let tail_start = s
        .char_indices()
        .nth_back(keep - 1)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let omitted = s.len() - head_end - (s.len() - tail_start);
    std::borrow::Cow::Owned(format!(
        "{}...({} bytes omitted)...{}",
        &s[..head_end],
        omitted,
        &s[tail_start..]
    ))
}

/// sensitive-logs 模式下输出完整请求体，但截断 base64 图片数据
///
/// 图片 base64 数据对诊断 400 错误没有价值，但可能占几十 KB。
/// 扫描 `"bytes":"<base64...>"` 模式，将长 base64 替换为占位符。
#[cfg(feature = "sensitive-logs")]
fn truncate_base64_in_request_body(s: &str) -> std::borrow::Cow<'_, str> {
    const MARKER: &str = r#""bytes":""#;
    const MIN_BASE64_LEN: usize = 200;

    // 快速路径：没有 "bytes":" 就直接返回
    if !s.contains(MARKER) {
        return std::borrow::Cow::Borrowed(s);
    }

    let mut result = String::with_capacity(s.len());
    let mut pos = 0;
    let bytes = s.as_bytes();

    while pos < bytes.len() {
        if let Some(offset) = s[pos..].find(MARKER) {
            let marker_start = pos + offset;
            let value_start = marker_start + MARKER.len();

            // 找到闭合引号（处理转义）
            let mut end = value_start;
            let mut escaped = false;
            while end < bytes.len() {
                if escaped {
                    escaped = false;
                    end += 1;
                    continue;
                }
                match bytes[end] {
                    b'\\' => {
                        escaped = true;
                        end += 1;
                    }
                    b'"' => break,
                    _ => end += 1,
                }
            }

            let value_len = end - value_start;
            if value_len >= MIN_BASE64_LEN && is_likely_base64(&s[value_start..end]) {
                result.push_str(&s[pos..value_start]);
                result.push_str(&format!("<BASE64_TRUNCATED:{}>", value_len));
                pos = end; // 跳到闭合引号，下一轮会输出它
            } else {
                // 不是 base64 或太短，原样保留
                result.push_str(&s[pos..value_start]);
                pos = value_start;
            }
        } else {
            result.push_str(&s[pos..]);
            break;
        }
    }

    std::borrow::Cow::Owned(result)
}

#[cfg(feature = "sensitive-logs")]
fn is_likely_base64(s: &str) -> bool {
    s.bytes()
        .take(100)
        .all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=')
}
