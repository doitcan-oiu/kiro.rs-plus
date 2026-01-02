//! Admin API 业务逻辑服务

use std::sync::Arc;

use crate::kiro::token_manager::MultiTokenManager;

use super::types::{BalanceResponse, CredentialStatusItem, CredentialsStatusResponse};

/// Admin 服务
///
/// 封装所有 Admin API 的业务逻辑
pub struct AdminService {
    token_manager: Arc<MultiTokenManager>,
}

impl AdminService {
    pub fn new(token_manager: Arc<MultiTokenManager>) -> Self {
        Self { token_manager }
    }

    /// 获取所有凭据状态
    pub fn get_all_credentials(&self) -> CredentialsStatusResponse {
        let snapshot = self.token_manager.snapshot();

        let credentials: Vec<CredentialStatusItem> = snapshot
            .entries
            .into_iter()
            .map(|entry| CredentialStatusItem {
                index: entry.index,
                priority: entry.priority,
                disabled: entry.disabled,
                failure_count: entry.failure_count,
                is_current: entry.index == snapshot.current_index,
                expires_at: entry.expires_at,
                auth_method: entry.auth_method,
                has_profile_arn: entry.has_profile_arn,
            })
            .collect();

        CredentialsStatusResponse {
            total: snapshot.total,
            available: snapshot.available,
            current_index: snapshot.current_index,
            credentials,
        }
    }

    /// 设置凭据禁用状态
    pub fn set_disabled(&self, index: usize, disabled: bool) -> anyhow::Result<()> {
        // 先获取当前凭据索引，用于判断是否需要切换
        let current_index = self.token_manager.snapshot().current_index;

        self.token_manager.set_disabled(index, disabled)?;

        // 只有禁用的是当前凭据时才尝试切换到下一个
        if disabled && index == current_index {
            let _ = self.token_manager.switch_to_next();
        }
        Ok(())
    }

    /// 设置凭据优先级
    pub fn set_priority(&self, index: usize, priority: u32) -> anyhow::Result<()> {
        self.token_manager.set_priority(index, priority)
    }

    /// 重置失败计数并重新启用
    pub fn reset_and_enable(&self, index: usize) -> anyhow::Result<()> {
        self.token_manager.reset_and_enable(index)
    }

    /// 获取凭据余额
    pub async fn get_balance(&self, index: usize) -> anyhow::Result<BalanceResponse> {
        let usage = self.token_manager.get_usage_limits_for(index).await?;

        let current_usage = usage.current_usage();
        let usage_limit = usage.usage_limit();
        let remaining = (usage_limit - current_usage).max(0.0);
        let usage_percentage = if usage_limit > 0.0 {
            (current_usage / usage_limit * 100.0).min(100.0)
        } else {
            0.0
        };

        Ok(BalanceResponse {
            index,
            subscription_title: usage.subscription_title().map(|s| s.to_string()),
            current_usage,
            usage_limit,
            remaining,
            usage_percentage,
            next_reset_at: usage.next_date_reset,
        })
    }
}
