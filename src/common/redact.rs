//! 日志脱敏工具
//!
//! 目标：避免在日志中输出敏感信息（Token、密钥、密码等）。

#![allow(dead_code)] // 工具模块，函数将在后续被调用

/// 统一的脱敏占位符
pub const REDACTED: &str = "<redacted>";

/// 将 `Option<String>` 映射为“是否存在”的脱敏表示：
/// - `Some(_)` -> `Some("<redacted>")`
/// - `None` -> `None`
#[inline]
pub fn redact_opt_string(value: &Option<String>) -> Option<&'static str> {
    value.as_ref().map(|_| REDACTED)
}

/// 脱敏邮箱：
/// - `abc@example.com` -> `a***@example.com`
/// - 无法解析时返回 `<redacted>`
pub fn mask_email(email: &str) -> String {
    let (local, domain) = match email.split_once('@') {
        Some((l, d)) if !l.is_empty() && !d.is_empty() => (l, d),
        _ => return REDACTED.to_string(),
    };

    // 保留首个完整字符（支持多字节 UTF-8）
    let first_char_end = local
        .char_indices()
        .nth(1)
        .map(|(i, _)| i)
        .unwrap_or(local.len());
    format!("{}***@{}", &local[..first_char_end], domain)
}

/// 脱敏 AWS ARN 中的 account id（第 5 段）：
/// `arn:aws:service:region:123456789012:resource` -> `arn:aws:service:region:***:resource`
pub fn mask_aws_account_id_in_arn(arn: &str) -> String {
    let mut parts = arn.splitn(6, ':').collect::<Vec<_>>();
    if parts.len() != 6 || parts[0] != "arn" {
        return arn.to_string();
    }

    if !parts[4].is_empty() {
        parts[4] = "***";
    }

    parts.join(":")
}

/// 脱敏 URL 中的 userinfo（仅当包含 `scheme://...@`）：
/// - `http://user:pass@host:port` -> `http://user:***@host:port`
/// - `http://user@host` -> `http://***@host`
pub fn mask_url_userinfo(url: &str) -> String {
    let Some(scheme_end) = url.find("://").map(|i| i + 3) else {
        return url.to_string();
    };

    // authority 结束于首个 '/'、'?'、'#' 或字符串末尾
    let authority_end = url[scheme_end..]
        .find(['/', '?', '#'])
        .map(|i| scheme_end + i)
        .unwrap_or(url.len());

    // '@' 必须在 authority 内才是 userinfo 分隔符
    let Some(at_pos) = url[scheme_end..authority_end]
        .find('@')
        .map(|i| scheme_end + i)
    else {
        return url.to_string();
    };

    let userinfo = &url[scheme_end..at_pos];
    if userinfo.is_empty() {
        return url.to_string();
    }

    let masked_userinfo = match userinfo.split_once(':') {
        Some((user, _pass)) if !user.is_empty() => format!("{}:***", user),
        _ => "***".to_string(),
    };

    format!(
        "{}{}{}",
        &url[..scheme_end],
        masked_userinfo,
        &url[at_pos..]
    )
}

/// 脱敏 User-Agent 中的 machine_id（常见形态为以 `-<machine_id>` 结尾）。
pub fn mask_user_agent_machine_id(value: &str) -> String {
    let Some(pos) = value.rfind('-') else {
        return value.to_string();
    };
    format!("{}{}", &value[..(pos + 1)], REDACTED)
}
