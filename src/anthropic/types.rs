//! Anthropic API 类型定义

use serde::{Deserialize, Serialize};

// === 错误响应 ===

/// API 错误响应
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// 错误详情
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl ErrorResponse {
    /// 创建新的错误响应
    pub fn new(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                error_type: error_type.into(),
                message: message.into(),
            },
        }
    }

    /// 创建认证错误响应
    pub fn authentication_error() -> Self {
        Self::new("authentication_error", "Invalid API key")
    }
}

// === Models 端点类型 ===

/// 模型信息
#[derive(Debug, Serialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub display_name: String,
    #[serde(rename = "type")]
    pub model_type: String,
    pub max_tokens: i32,
}

/// 模型列表响应
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

// === Messages 端点类型 ===

/// 最大思考预算 tokens
const MAX_BUDGET_TOKENS: i32 = 24576;

/// Thinking 配置
#[derive(Debug, Deserialize, Clone)]
pub struct Thinking {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(
        default = "default_budget_tokens",
        deserialize_with = "deserialize_budget_tokens"
    )]
    pub budget_tokens: i32,
}

fn default_budget_tokens() -> i32 {
    20000
}
fn deserialize_budget_tokens<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = i32::deserialize(deserializer)?;
    Ok(value.min(MAX_BUDGET_TOKENS))
}

/// Claude Code 请求中的 metadata
#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    /// 用户 ID，格式如: user_xxx_account__session_0b4445e1-f5be-49e1-87ce-62bbc28ad705
    pub user_id: Option<String>,
}

/// Messages 请求体
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    pub system: Option<Vec<SystemMessage>>,
    /// tools 可以是普通 Tool 或 WebSearchTool 等多种格式，使用 Value 灵活处理
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: Option<serde_json::Value>,
    pub thinking: Option<Thinking>,
    /// Claude Code 请求中的 metadata，包含 session 信息
    pub metadata: Option<Metadata>,
}

fn default_max_tokens() -> i32 {
    4096
}

/// 消息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    /// 可以是 string 或 ContentBlock 数组
    pub content: serde_json::Value,
}

/// 系统消息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemMessage {
    #[serde(rename = "type", default = "default_message_type")]
    pub message_type: String,
    pub text: String,
}

fn default_message_type() -> String {
    "text".to_string()
}

/// 内容块
#[derive(Debug, Deserialize, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ImageSource>,
}

/// 图片数据源
#[derive(Debug, Deserialize, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

// === Count Tokens 端点类型 ===

/// Token 计数请求
#[derive(Debug, Serialize, Deserialize)]
pub struct CountTokensRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// Token 计数响应
#[derive(Debug, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 new-api 发送的完整 Claude 请求格式
    /// 包含：system 数组格式、普通 Tool、WebSearchTool、omitempty 字段缺失等情况
    #[test]
    fn test_new_api_claude_request_format() {
        // 模拟 new-api 发送的真实请求
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "system": [
                {"type": "text", "text": "You are a helpful assistant"}
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                },
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                }
            ],
            "stream": true
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).expect("应该能解析 new-api 请求");

        assert_eq!(req.model, "claude-sonnet-4-5-20250929");
        assert_eq!(req.max_tokens, 4096); // 默认值
        assert!(req.stream);
        assert_eq!(req.messages.len(), 1);

        // 验证 system
        let system = req.system.expect("应该有 system");
        assert_eq!(system.len(), 1);
        assert_eq!(system[0].message_type, "text");
        assert_eq!(system[0].text, "You are a helpful assistant");

        // 验证 tools（包含普通 Tool 和 WebSearchTool）
        let tools = req.tools.expect("应该有 tools");
        assert_eq!(tools.len(), 2);
        assert_eq!(
            tools[0].get("name").unwrap().as_str().unwrap(),
            "get_weather"
        );
        assert_eq!(
            tools[1].get("type").unwrap().as_str().unwrap(),
            "web_search_20250305"
        );
    }

    /// 测试 max_tokens 缺失时使用默认值
    #[test]
    fn test_max_tokens_default() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 4096);
    }

    /// 测试 system 中 type 字段缺失时使用默认值
    #[test]
    fn test_system_message_type_default() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": [{"text": "Be helpful"}]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let system = req.system.unwrap();
        assert_eq!(system[0].message_type, "text");
    }

    /// 测试 Tool 的 description 可选
    #[test]
    fn test_tool_description_optional() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "my_tool",
                "input_schema": {"type": "object"}
            }]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.unwrap();
        assert!(tools[0].get("description").is_none());
    }

    /// 测试 SystemMessage 序列化时 type 字段存在
    #[test]
    fn test_system_message_serialization() {
        let msg = SystemMessage {
            message_type: "text".to_string(),
            text: "Hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"text""#));
    }

    /// 测试复杂消息内容（数组格式）
    #[test]
    fn test_complex_message_content() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
                ]
            }]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages[0].content.is_array());
    }

    /// 测试 thinking 配置
    #[test]
    fn test_thinking_config() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10000
            }
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let thinking = req.thinking.unwrap();
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, 10000);
    }

    /// 测试 thinking budget_tokens 超过最大值时被截断
    #[test]
    fn test_thinking_budget_tokens_capped() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {
                "type": "enabled",
                "budget_tokens": 100000
            }
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let thinking = req.thinking.unwrap();
        assert_eq!(thinking.budget_tokens, 24576); // MAX_BUDGET_TOKENS
    }

    // ==================== new-api 兼容性测试 ====================

    /// 测试 new-api 转换后的 tool_result 格式
    /// new-api 将 OpenAI 的 role:"tool" 转换为 role:"user" + tool_result content block
    #[test]
    fn test_new_api_tool_result_format() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check the weather."},
                    {"type": "tool_use", "id": "toolu_01ABC", "name": "get_weather", "input": {"location": "Tokyo"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": "Sunny, 25°C"}
                ]}
            ]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).expect("应该能解析 tool_result 格式");
        assert_eq!(req.messages.len(), 3);

        // 验证 assistant 消息包含 tool_use
        let assistant_content = req.messages[1].content.as_array().unwrap();
        assert_eq!(assistant_content.len(), 2);
        assert_eq!(assistant_content[1].get("type").unwrap(), "tool_use");
        assert_eq!(assistant_content[1].get("id").unwrap(), "toolu_01ABC");
        assert_eq!(assistant_content[1].get("name").unwrap(), "get_weather");

        // 验证 user 消息包含 tool_result
        let user_content = req.messages[2].content.as_array().unwrap();
        assert_eq!(user_content[0].get("type").unwrap(), "tool_result");
        assert_eq!(user_content[0].get("tool_use_id").unwrap(), "toolu_01ABC");
    }

    /// 测试 new-api 转换后的占位符消息 "..."
    /// new-api 会将空 content 替换为 "..."，首条非 user 消息前也会插入 "..."
    #[test]
    fn test_new_api_placeholder_content() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "Hello!"}
            ]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).expect("应该能解析占位符消息");
        assert_eq!(req.messages[0].content.as_str().unwrap(), "...");
    }

    /// 测试 new-api 转换后的 tool_choice 格式
    #[test]
    fn test_new_api_tool_choice_formats() {
        // auto 格式
        let json_auto = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "test", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"}
        }"#;
        let req: MessagesRequest = serde_json::from_str(json_auto).unwrap();
        assert_eq!(req.tool_choice.unwrap().get("type").unwrap(), "auto");

        // any 格式 (OpenAI "required" 转换而来)
        let json_any = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "test", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"}
        }"#;
        let req: MessagesRequest = serde_json::from_str(json_any).unwrap();
        assert_eq!(req.tool_choice.unwrap().get("type").unwrap(), "any");

        // tool 格式 (指定具体工具)
        let json_tool = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }"#;
        let req: MessagesRequest = serde_json::from_str(json_tool).unwrap();
        let tool_choice = req.tool_choice.unwrap();
        assert_eq!(tool_choice.get("type").unwrap(), "tool");
        assert_eq!(tool_choice.get("name").unwrap(), "get_weather");
    }

    /// 测试 new-api 转换后的 tool_choice 带 disable_parallel_tool_use
    #[test]
    fn test_new_api_tool_choice_parallel_disabled() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "test", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let tool_choice = req.tool_choice.unwrap();
        assert_eq!(tool_choice.get("disable_parallel_tool_use").unwrap(), true);
    }

    /// 测试 new-api 转换后的多个 tool_result 合并到同一 user 消息
    #[test]
    fn test_new_api_multiple_tool_results_merged() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "user", "content": "Check weather and time"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_weather", "name": "get_weather", "input": {}},
                    {"type": "tool_use", "id": "toolu_time", "name": "get_time", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_weather", "content": "Sunny"},
                    {"type": "tool_result", "tool_use_id": "toolu_time", "content": "10:00 AM"}
                ]}
            ]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).expect("应该能解析多个 tool_result");

        // 验证多个 tool_result 在同一个 user 消息中
        let user_content = req.messages[2].content.as_array().unwrap();
        assert_eq!(user_content.len(), 2);
        assert_eq!(user_content[0].get("tool_use_id").unwrap(), "toolu_weather");
        assert_eq!(user_content[1].get("tool_use_id").unwrap(), "toolu_time");
    }

    /// 测试 new-api 转换后的 tool_result 带 is_error 标记
    #[test]
    fn test_new_api_tool_result_with_error() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "risky_action", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "Error: Permission denied", "is_error": true}
                ]}
            ]
        }"#;

        let req: MessagesRequest =
            serde_json::from_str(json).expect("应该能解析带错误的 tool_result");
        let user_content = req.messages[2].content.as_array().unwrap();
        assert_eq!(user_content[0].get("is_error").unwrap(), true);
    }

    /// 测试 new-api 转换后的图片消息格式 (base64)
    #[test]
    fn test_new_api_image_base64_format() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "/9j/4AAQSkZJRg..."
                        }
                    }
                ]
            }]
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).expect("应该能解析图片消息");
        let content = req.messages[0].content.as_array().unwrap();
        assert_eq!(content.len(), 2);

        let image_block = &content[1];
        assert_eq!(image_block.get("type").unwrap(), "image");
        let source = image_block.get("source").unwrap();
        assert_eq!(source.get("type").unwrap(), "base64");
        assert_eq!(source.get("media_type").unwrap(), "image/jpeg");
    }

    /// 测试 new-api 转换后的 metadata 字段
    #[test]
    fn test_new_api_metadata_field() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "metadata": {
                "user_id": "user_abc123_account__session_0b4445e1-f5be-49e1-87ce-62bbc28ad705"
            }
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).expect("应该能解析 metadata");
        let metadata = req.metadata.unwrap();
        assert!(metadata.user_id.unwrap().starts_with("user_"));
    }

    /// 测试 new-api 转换后的完整多轮对话（包含工具调用）
    #[test]
    fn test_new_api_full_conversation_with_tools() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 8192,
            "system": [
                {"type": "text", "text": "You are a helpful assistant with access to tools."}
            ],
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I'll check the weather for you."},
                    {"type": "tool_use", "id": "toolu_01XYZ", "name": "get_weather", "input": {"location": "Tokyo", "unit": "celsius"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01XYZ", "content": "{\"temperature\": 22, \"condition\": \"Partly cloudy\"}"}
                ]},
                {"role": "assistant", "content": "The weather in Tokyo is 22°C and partly cloudy."},
                {"role": "user", "content": "Thanks!"}
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            ],
            "stream": true
        }"#;

        let req: MessagesRequest = serde_json::from_str(json).expect("应该能解析完整对话");

        // 验证基本字段
        assert_eq!(req.model, "claude-sonnet-4-5-20250929");
        assert_eq!(req.max_tokens, 8192);
        assert!(req.stream);

        // 验证消息数量和角色交替
        assert_eq!(req.messages.len(), 5);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[1].role, "assistant");
        assert_eq!(req.messages[2].role, "user"); // tool_result
        assert_eq!(req.messages[3].role, "assistant");
        assert_eq!(req.messages[4].role, "user");

        // 验证 tool_use 中的 input 是对象而非字符串
        let assistant_content = req.messages[1].content.as_array().unwrap();
        let tool_use = &assistant_content[1];
        let input = tool_use.get("input").unwrap();
        assert!(input.is_object());
        assert_eq!(input.get("location").unwrap(), "Tokyo");
    }

    /// 测试 new-api 转换后的 stop_sequences 格式
    #[test]
    fn test_new_api_stop_sequences() {
        let json = r#"{
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["Human:", "Assistant:"]
        }"#;

        // 注意：当前 MessagesRequest 可能没有 stop_sequences 字段
        // 如果需要支持，需要添加该字段
        let result: Result<MessagesRequest, _> = serde_json::from_str(json);
        // 即使没有该字段，serde 默认会忽略未知字段，不会报错
        assert!(result.is_ok());
    }
}
