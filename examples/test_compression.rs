use serde_json::Value;
use std::fs;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| {
        anyhow::anyhow!("Usage: cargo run --example test_compression -- <path-to-json>")
    })?;

    let content = fs::read_to_string(&path)?;
    let data: Value = serde_json::from_str(&content)?;

    let (req, raw_body_len) = if let Some(body_str) = data["request_body"].as_str() {
        (
            serde_json::from_str::<Value>(body_str)?,
            Some(body_str.len()),
        )
    } else {
        (data, None)
    };

    let body_len = match raw_body_len {
        Some(n) => n,
        None => serde_json::to_string(&req).unwrap_or_default().len(),
    };
    println!(
        "原始请求体大小: {} bytes ({:.1} KB)",
        body_len,
        body_len as f64 / 1024.0
    );

    if let Some(messages) = req["messages"].as_array() {
        println!("消息数量: {}", messages.len());

        let mut user_count = 0;
        let mut assistant_count = 0;
        let mut total_chars = 0;
        let mut tool_result_chars = 0;
        let mut tool_use_chars = 0;

        for msg in messages {
            match msg["role"].as_str() {
                Some("user") => user_count += 1,
                Some("assistant") => assistant_count += 1,
                _ => {}
            }

            if let Some(content) = msg["content"].as_array() {
                for item in content {
                    if let Some(text) = item["text"].as_str() {
                        total_chars += text.len();
                    }
                    // 统计 tool_result
                    if item["type"].as_str() == Some("tool_result")
                        && let Some(result_content) = item["content"].as_array()
                    {
                        for result_item in result_content {
                            if let Some(text) = result_item["text"].as_str() {
                                tool_result_chars += text.len();
                            }
                        }
                    }
                    // 统计 tool_use
                    if item["type"].as_str() == Some("tool_use") {
                        let input_str =
                            serde_json::to_string(&item["input"]).unwrap_or_else(|_| "null".into());
                        tool_use_chars += input_str.len();
                    }
                }
            }
        }

        println!("  - user: {}", user_count);
        println!("  - assistant: {}", assistant_count);
        println!(
            "  - 文本字符数: {} ({:.1} KB)",
            total_chars,
            total_chars as f64 / 1024.0
        );
        println!(
            "  - tool_result 字符数: {} ({:.1} KB)",
            tool_result_chars,
            tool_result_chars as f64 / 1024.0
        );
        println!(
            "  - tool_use input 字符数: {} ({:.1} KB)",
            tool_use_chars,
            tool_use_chars as f64 / 1024.0
        );

        // 模拟历史截断
        let max_history_turns = 80;
        let max_history_chars = 400_000;

        let turns = messages.len() / 2;
        println!("\n压缩模拟（默认配置）:");
        println!("  - 当前轮数: {} (阈值: {})", turns, max_history_turns);

        if turns > max_history_turns {
            let to_remove = turns - max_history_turns;
            println!("  - 需要移除: {} 轮 ({} 条消息)", to_remove, to_remove * 2);
        } else {
            println!("  - 轮数未超限");
        }

        let total_content_chars = total_chars + tool_result_chars + tool_use_chars;
        println!(
            "  - 总内容字符数: {} ({:.1} KB)",
            total_content_chars,
            total_content_chars as f64 / 1024.0
        );

        if total_content_chars > max_history_chars {
            println!(
                "  - 字符数超限: {} > {}",
                total_content_chars, max_history_chars
            );
        } else {
            println!("  - 字符数未超限");
        }
    }

    if let Some(tools) = req["tools"].as_array() {
        let tools_str = serde_json::to_string(tools).unwrap_or_default();
        println!("\n工具数量: {}", tools.len());
        println!(
            "工具定义总大小: {} bytes ({:.1} KB)",
            tools_str.len(),
            tools_str.len() as f64 / 1024.0
        );

        // 统计每个工具描述的大小
        let mut total_desc_chars = 0;
        for tool in tools {
            if let Some(desc) = tool["description"].as_str() {
                total_desc_chars += desc.len();
            }
        }
        println!(
            "工具描述总字符数: {} ({:.1} KB)",
            total_desc_chars,
            total_desc_chars as f64 / 1024.0
        );
    }

    Ok(())
}
