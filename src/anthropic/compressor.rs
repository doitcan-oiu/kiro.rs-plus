//! 输入压缩管道
//!
//! 在协议转换完成后、发送到上游前，对 `ConversationState` 执行多层压缩，
//! 以规避 Kiro 上游 ~400KB 请求体大小限制。
//!
//! 压缩顺序（低风险 → 高风险）：
//! 1. 空白压缩
//! 2. thinking 块丢弃/截断
//! 3. tool_result 智能截断
//! 4. tool_use input 截断
//! 5. 历史截断

use crate::kiro::model::requests::conversation::{ConversationState, Message};
use crate::model::config::CompressionConfig;

/// 压缩统计信息
#[derive(Debug, Default)]
pub struct CompressionStats {
    pub whitespace_saved: usize,
    pub thinking_saved: usize,
    pub tool_result_saved: usize,
    pub tool_use_input_saved: usize,
    pub history_turns_removed: usize,
}

impl CompressionStats {
    /// 总节省字节数
    pub fn total_saved(&self) -> usize {
        self.whitespace_saved
            + self.thinking_saved
            + self.tool_result_saved
            + self.tool_use_input_saved
    }
}

/// 压缩管道入口
///
/// 按顺序执行各层压缩，返回统计信息。
pub fn compress(state: &mut ConversationState, config: &CompressionConfig) -> CompressionStats {
    let mut stats = CompressionStats::default();

    if !config.enabled {
        return stats;
    }

    // 1. 空白压缩
    if config.whitespace_compression {
        stats.whitespace_saved = compress_whitespace_pass(state);
    }

    // 2. thinking 丢弃/截断
    if config.thinking_strategy != "keep" {
        stats.thinking_saved = compress_thinking_pass(state, &config.thinking_strategy);
    }

    // 3. tool_result 智能截断
    if config.tool_result_max_chars > 0 {
        stats.tool_result_saved = compress_tool_results_pass(
            state,
            config.tool_result_max_chars,
            config.tool_result_head_lines,
            config.tool_result_tail_lines,
        );
    }

    // 4. tool_use input 截断
    if config.tool_use_input_max_chars > 0 {
        stats.tool_use_input_saved =
            compress_tool_use_inputs_pass(state, config.tool_use_input_max_chars);
    }

    // 5. 历史截断（最后手段）
    if config.max_history_turns > 0 || config.max_history_chars > 0 {
        stats.history_turns_removed =
            compress_history_pass(state, config.max_history_turns, config.max_history_chars);
    }

    stats
}

// ============ 空白压缩 ============

/// 空白压缩：连续空行(3+)→单空行，行尾空格移除，保留行首缩进
fn compress_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut consecutive_empty = 0u32;

    for line in text.split('\n') {
        let trimmed_end = line.trim_end();

        if trimmed_end.is_empty() {
            consecutive_empty += 1;
            if consecutive_empty <= 2 && !result.is_empty() {
                result.push('\n');
            }
        } else {
            consecutive_empty = 0;
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(trimmed_end);
        }
    }

    result
}

/// 对 ConversationState 中所有文本字段执行空白压缩
fn compress_whitespace_pass(state: &mut ConversationState) -> usize {
    let mut saved = 0usize;

    for msg in &mut state.history {
        match msg {
            Message::User(user_msg) => {
                saved += compress_string_field(&mut user_msg.user_input_message.content);
            }
            Message::Assistant(assistant_msg) => {
                saved +=
                    compress_string_field(&mut assistant_msg.assistant_response_message.content);
            }
        }
    }

    saved += compress_string_field(&mut state.current_message.user_input_message.content);
    saved
}

/// 压缩单个字符串字段，返回节省的字节数
///
/// 跳过仅为空格占位符 " " 的字段（Kiro API 要求 content 不能为空，
/// converter 使用 " " 作为占位符）
fn compress_string_field(field: &mut String) -> usize {
    if field == " " {
        return 0;
    }
    let original_len = field.len();
    let compressed = compress_whitespace(field);
    if compressed.len() < original_len {
        let saved = original_len - compressed.len();
        *field = compressed;
        saved
    } else {
        0
    }
}

// ============ thinking 压缩 ============

/// 处理 history 中 assistant 消息的 `<thinking>...</thinking>` 块
fn compress_thinking_pass(state: &mut ConversationState, strategy: &str) -> usize {
    let mut saved = 0usize;

    for msg in &mut state.history {
        if let Message::Assistant(assistant_msg) = msg {
            let content = &mut assistant_msg.assistant_response_message.content;
            let original_len = content.len();

            match strategy {
                "discard" => *content = remove_thinking_blocks(content),
                "truncate" => *content = truncate_thinking_blocks(content, 500),
                _ => {}
            }

            if content.len() < original_len {
                saved += original_len - content.len();
            }
        }
    }

    saved
}

/// 移除所有 `<thinking>...</thinking>` 块
fn remove_thinking_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(start) = remaining.find("<thinking>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</thinking>") {
            remaining = &remaining[start + end + "</thinking>".len()..];
        } else {
            remaining = "";
        }
    }
    result.push_str(remaining);
    result.trim().to_string()
}

/// 截断 `<thinking>...</thinking>` 块内容，保留前 N 个字符
fn truncate_thinking_blocks(text: &str, max_chars: usize) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(start) = remaining.find("<thinking>") {
        result.push_str(&remaining[..start]);
        let after_tag = &remaining[start + "<thinking>".len()..];

        if let Some(end) = after_tag.find("</thinking>") {
            let thinking_content = &after_tag[..end];
            let truncated = safe_char_truncate(thinking_content, max_chars);
            result.push_str("<thinking>");
            result.push_str(truncated);
            if truncated.len() < thinking_content.len() {
                result.push_str("...[truncated]");
            }
            result.push_str("</thinking>");
            remaining = &after_tag[end + "</thinking>".len()..];
        } else {
            let truncated = safe_char_truncate(after_tag, max_chars);
            result.push_str("<thinking>");
            result.push_str(truncated);
            result.push_str("...[truncated]</thinking>");
            remaining = "";
        }
    }
    result.push_str(remaining);
    result
}

// ============ tool_result 智能截断 ============

/// 按行智能截断，保留头尾行
fn smart_truncate_by_lines(
    text: &str,
    max_chars: usize,
    head_lines: usize,
    tail_lines: usize,
) -> (String, usize) {
    if text.len() <= max_chars {
        return (text.to_string(), 0);
    }

    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len();

    if total_lines <= head_lines + tail_lines {
        let half = max_chars / 2;
        let head = safe_char_truncate(text, half);
        let tail_chars = max_chars.saturating_sub(head.len());
        let tail_start = text
            .char_indices()
            .rev()
            .nth(tail_chars.saturating_sub(1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let tail = &text[tail_start..];
        let omitted = text.len().saturating_sub(head.len() + tail.len());
        let result = format!("{}\n... [{} chars omitted] ...\n{}", head, omitted, tail);
        let saved = text.len().saturating_sub(result.len());
        return (result, saved);
    }

    let head_part: String = lines[..head_lines].join("\n");
    let tail_part: String = lines[total_lines - tail_lines..].join("\n");
    let omitted_lines = total_lines - head_lines - tail_lines;
    let omitted_chars = text.len().saturating_sub(head_part.len() + tail_part.len());

    let result = format!(
        "{}\n... [{} lines omitted ({} chars)] ...\n{}",
        head_part, omitted_lines, omitted_chars, tail_part
    );
    let saved = text.len().saturating_sub(result.len());
    (result, saved)
}

/// 遍历所有 tool_result 的 text 字段，执行智能截断
fn compress_tool_results_pass(
    state: &mut ConversationState,
    max_chars: usize,
    head_lines: usize,
    tail_lines: usize,
) -> usize {
    let mut saved = 0usize;

    for msg in &mut state.history {
        if let Message::User(user_msg) = msg {
            for result in &mut user_msg
                .user_input_message
                .user_input_message_context
                .tool_results
            {
                saved += truncate_tool_result_content(
                    &mut result.content,
                    max_chars,
                    head_lines,
                    tail_lines,
                );
            }
        }
    }

    for result in &mut state
        .current_message
        .user_input_message
        .user_input_message_context
        .tool_results
    {
        saved +=
            truncate_tool_result_content(&mut result.content, max_chars, head_lines, tail_lines);
    }

    saved
}

/// 截断单个 tool_result 的 content 数组中的 text 字段
fn truncate_tool_result_content(
    content: &mut [serde_json::Map<String, serde_json::Value>],
    max_chars: usize,
    head_lines: usize,
    tail_lines: usize,
) -> usize {
    let mut saved = 0usize;

    for map in content.iter_mut() {
        if let Some(serde_json::Value::String(text)) = map.get_mut("text") {
            if text.len() > max_chars {
                let (truncated, s) =
                    smart_truncate_by_lines(text, max_chars, head_lines, tail_lines);
                saved += s;
                *text = truncated;
            }
        }
    }

    saved
}


// ============ tool_use input 截断 ============

/// 遍历 history 中 assistant 消息的 tool_use input，截断大字符串字段
fn compress_tool_use_inputs_pass(state: &mut ConversationState, max_chars: usize) -> usize {
    let mut saved = 0usize;

    for msg in &mut state.history {
        if let Message::Assistant(assistant_msg) = msg {
            if let Some(ref mut tool_uses) = assistant_msg.assistant_response_message.tool_uses {
                for tool_use in tool_uses.iter_mut() {
                    let serialized = serde_json::to_string(&tool_use.input).unwrap_or_default();
                    if serialized.len() > max_chars {
                        saved += truncate_json_value_strings(&mut tool_use.input, max_chars);
                    }
                }
            }
        }
    }

    saved
}

/// 递归截断 JSON 值中的大字符串
fn truncate_json_value_strings(value: &mut serde_json::Value, max_chars: usize) -> usize {
    let mut saved = 0usize;

    match value {
        serde_json::Value::String(s) => {
            if s.len() > max_chars {
                let original_len = s.len();
                let truncated = safe_char_truncate(s, max_chars);
                *s = format!(
                    "{}...[truncated {} chars]",
                    truncated,
                    original_len - truncated.len()
                );
                saved += original_len.saturating_sub(s.len());
            }
        }
        serde_json::Value::Object(map) => {
            for (_, v) in map.iter_mut() {
                saved += truncate_json_value_strings(v, max_chars);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr.iter_mut() {
                saved += truncate_json_value_strings(v, max_chars);
            }
        }
        _ => {}
    }

    saved
}


// ============ 历史截断 ============

/// 历史截断：保留前 2 条（系统消息对），从前往后成对移除
fn compress_history_pass(
    state: &mut ConversationState,
    max_turns: usize,
    max_chars: usize,
) -> usize {
    let mut removed = 0usize;
    let preserve_count = 2;

    // 按轮数截断
    if max_turns > 0 {
        let max_messages = preserve_count + max_turns * 2;
        while state.history.len() > max_messages && state.history.len() > preserve_count + 2 {
            state.history.remove(preserve_count);
            state.history.remove(preserve_count);
            removed += 1;
        }
    }

    // 按字符数截断
    if max_chars > 0 {
        loop {
            let total_chars: usize = state
                .history
                .iter()
                .map(|msg| match msg {
                    Message::User(u) => u.user_input_message.content.len(),
                    Message::Assistant(a) => a.assistant_response_message.content.len(),
                })
                .sum();

            if total_chars <= max_chars || state.history.len() <= preserve_count + 2 {
                break;
            }

            state.history.remove(preserve_count);
            state.history.remove(preserve_count);
            removed += 1;
        }
    }

    removed
}

// ============ 工具函数 ============

/// 安全 UTF-8 字符截断
fn safe_char_truncate(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::requests::conversation::*;
    use crate::kiro::model::requests::tool::{ToolResult, ToolUseEntry};
    use crate::model::config::CompressionConfig;

    fn make_simple_state(history_content: Vec<(&str, &str)>, current: &str) -> ConversationState {
        let mut history = Vec::new();
        for (user, assistant) in history_content {
            history.push(Message::User(HistoryUserMessage::new(
                user,
                "claude-sonnet-4.5",
            )));
            history.push(Message::Assistant(HistoryAssistantMessage::new(assistant)));
        }
        ConversationState::new("test-conv")
            .with_current_message(CurrentMessage::new(UserInputMessage::new(
                current,
                "claude-sonnet-4.5",
            )))
            .with_history(history)
    }

    #[test]
    fn test_compress_whitespace_consecutive_empty_lines() {
        let input = "line1\n\n\n\n\nline2";
        let result = compress_whitespace(input);
        // 5 个空行 → 保留最多 2 个（即 line1 + 2 个 \n + line2）
        assert_eq!(result, "line1\n\n\nline2");
    }

    #[test]
    fn test_compress_whitespace_trailing_spaces() {
        let input = "hello   \nworld  ";
        let result = compress_whitespace(input);
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_compress_whitespace_preserves_indentation() {
        let input = "    indented\n        more indented";
        let result = compress_whitespace(input);
        assert_eq!(result, "    indented\n        more indented");
    }

    #[test]
    fn test_smart_truncate_short_content_unchanged() {
        let input = "short text";
        let (result, saved) = smart_truncate_by_lines(input, 100, 5, 3);
        assert_eq!(result, input);
        assert_eq!(saved, 0);
    }

    #[test]
    fn test_smart_truncate_preserves_head_tail() {
        let lines: Vec<String> = (0..200).map(|i| format!("line {}", i)).collect();
        let input = lines.join("\n");
        let (result, _saved) = smart_truncate_by_lines(&input, 100, 3, 2);
        assert!(result.starts_with("line 0\nline 1\nline 2\n"));
        assert!(result.ends_with("line 198\nline 199"));
        assert!(result.contains("lines omitted"));
    }

    #[test]
    fn test_safe_char_truncate_utf8() {
        let input = "你好世界abcd";
        let result = safe_char_truncate(input, 4);
        assert_eq!(result, "你好世界");
    }

    #[test]
    fn test_thinking_discard() {
        let mut state = make_simple_state(
            vec![(
                "hi",
                "<thinking>long thinking content here</thinking>\n\nactual response",
            )],
            "next",
        );
        let config = CompressionConfig {
            thinking_strategy: "discard".to_string(),
            ..Default::default()
        };
        let stats = compress(&mut state, &config);
        assert!(stats.thinking_saved > 0);
        // assistant content 不应包含 thinking 标签
        if let Message::Assistant(a) = &state.history[1] {
            assert!(!a.assistant_response_message.content.contains("<thinking>"));
            assert!(
                a.assistant_response_message
                    .content
                    .contains("actual response")
            );
        }
    }

    #[test]
    fn test_thinking_truncate() {
        let long_thinking = "a".repeat(1000);
        let content = format!("<thinking>{}</thinking>\n\nresponse", long_thinking);
        let mut state = make_simple_state(vec![("hi", &content)], "next");
        let config = CompressionConfig {
            thinking_strategy: "truncate".to_string(),
            ..Default::default()
        };
        let stats = compress(&mut state, &config);
        assert!(stats.thinking_saved > 0);
        if let Message::Assistant(a) = &state.history[1] {
            assert!(a.assistant_response_message.content.contains("<thinking>"));
            assert!(a.assistant_response_message.content.contains("[truncated]"));
        }
    }

    #[test]
    fn test_thinking_keep() {
        let content = "<thinking>keep me</thinking>\n\nresponse";
        let mut state = make_simple_state(vec![("hi", content)], "next");
        let config = CompressionConfig {
            thinking_strategy: "keep".to_string(),
            ..Default::default()
        };
        let stats = compress(&mut state, &config);
        assert_eq!(stats.thinking_saved, 0);
        if let Message::Assistant(a) = &state.history[1] {
            assert!(
                a.assistant_response_message
                    .content
                    .contains("<thinking>keep me</thinking>")
            );
        }
    }

    #[test]
    fn test_tool_result_truncation() {
        let long_text = "x\n".repeat(500);
        let mut state = ConversationState::new("test")
            .with_current_message(CurrentMessage::new(
                UserInputMessage::new("msg", "claude-sonnet-4.5").with_context(
                    UserInputMessageContext::new()
                        .with_tool_results(vec![ToolResult::success("t1", &long_text)]),
                ),
            ))
            .with_history(Vec::new());

        let config = CompressionConfig {
            tool_result_max_chars: 100,
            tool_result_head_lines: 3,
            tool_result_tail_lines: 2,
            ..Default::default()
        };
        let stats = compress(&mut state, &config);
        assert!(stats.tool_result_saved > 0);
    }

    #[test]
    fn test_tool_use_input_truncation() {
        let long_input = serde_json::json!({
            "content": "a".repeat(10000)
        });
        let mut assistant_msg = AssistantMessage::new("using tool");
        assistant_msg = assistant_msg.with_tool_uses(vec![
            ToolUseEntry::new("t1", "write").with_input(long_input),
        ]);

        let mut state = ConversationState::new("test")
            .with_current_message(CurrentMessage::new(UserInputMessage::new(
                "next",
                "claude-sonnet-4.5",
            )))
            .with_history(vec![
                Message::User(HistoryUserMessage::new("do it", "claude-sonnet-4.5")),
                Message::Assistant(HistoryAssistantMessage {
                    assistant_response_message: assistant_msg,
                }),
            ]);

        let config = CompressionConfig {
            tool_use_input_max_chars: 100,
            ..Default::default()
        };
        let stats = compress(&mut state, &config);
        assert!(stats.tool_use_input_saved > 0);
    }

    #[test]
    fn test_history_truncation_preserves_system_pair() {
        // 创建 system pair (2) + 5 轮对话 (10) = 12 条消息
        let mut history_content = vec![("system prompt", "I will follow these instructions.")];
        for _i in 0..5 {
            history_content.push(("user msg", "assistant msg"));
        }
        let mut state = make_simple_state(history_content, "current");

        let config = CompressionConfig {
            max_history_turns: 2,
            ..Default::default()
        };
        let stats = compress(&mut state, &config);
        assert!(stats.history_turns_removed > 0);
        // 应保留 system pair (2) + 2 轮 (4) = 6 条
        assert_eq!(state.history.len(), 6);
        // 第一对应该是 system pair
        if let Message::User(u) = &state.history[0] {
            assert!(u.user_input_message.content.contains("system prompt"));
        }
    }

    #[test]
    fn test_compress_disabled_no_change() {
        let content = "line1\n\n\n\n\nline2   ";
        let mut state = make_simple_state(vec![("hi", content)], "next");
        let original_content = content.to_string();

        let config = CompressionConfig {
            enabled: false,
            ..Default::default()
        };
        let stats = compress(&mut state, &config);
        assert_eq!(stats.total_saved(), 0);
        assert_eq!(stats.history_turns_removed, 0);
        // content 应保持不变
        if let Message::Assistant(a) = &state.history[1] {
            assert_eq!(a.assistant_response_message.content, original_content);
        }
    }
}
