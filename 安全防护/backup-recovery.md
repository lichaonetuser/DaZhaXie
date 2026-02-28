# Agent 状态备份与恢复方案

## 1. 备份目标

确保 Agent 可以在被带偏或发生异常时，快速恢复到之前的正常状态，最大程度减少损失。

---

## 2. 备份类型

### 2.1 定时快照（Automatic Snapshot）

按照固定时间间隔自动保存完整状态。

- **间隔**：默认5分钟
- **保留数量**：默认20个
- **存储内容**：系统Prompt + 最近50条消息 + 元数据

### 2.2 事件触发备份（Event-based Backup）

在特定事件发生时立即备份。

**触发条件：**
- 敏感词拦截
- Prompt完整性失败
- 行为异常
- 手动触发

### 2.3 完整镜像备份（Full Mirror Backup）

定期创建完整的系统镜像，包含所有配置和状态。

**备份内容：**
- protected/ 目录（系统Prompt）
- openclaw.json（配置）
- agents/ 目录（会话数据）

---

## 3. 恢复机制

### 3.1 自动恢复触发条件

| 条件 | 检测方法 | 动作 |
|------|----------|------|
| Prompt被篡改 | 哈希校验失败 | 自动回滚 |
| 频繁阻断 | 连续3次敏感词拦截 | 自动回滚 |
| 行为异常 | 风格偏离度>阈值 | 告警+可选回滚 |

### 3.2 恢复命令

```bash
# 恢复到最近快照
openclaw session restore --latest

# 恢复到指定时间
openclaw session restore --timestamp 2026-02-28_18-00-00

# 恢复到事件触发点
openclaw session restore --event blocked --latest

# 列出所有可用备份
openclaw session list-backups

# 手动创建备份
openclaw session backup --create "手动备份"

# 恢复完整镜像
openclaw session restore-mirror --latest
```

### 3.3 恢复流程

1. 检测到需要恢复
2. 检查备份可用性（快照/事件/镜像）
3. 加载备份数据并验证完整性
4. 执行恢复（Prompt + 消息 + 模型状态）
5. 验证恢复结果（哈希校验 + 测试响应）
6. 通知用户

---

## 4. 存储策略

### 4.1 多级存储

| 级别 | 位置 | 保留时间 | 用途 |
|------|------|----------|------|
| 热存储 | ~/.openclaw/backups | 24小时 | 快速恢复 |
| 温存储 | ~/.openclaw/archive | 7天 | 近期回溯 |
| 冷存储 | S3/OSS（可选） | 30天 | 长期存档 |

### 4.2 存储配额

```yaml
storage:
  hot:
    max_size_mb: 500
    max_count: 50
  warm:
    max_size_mb: 2048
    retention_days: 7
  cold:
    enabled: false
    retention_days: 30
```

---

## 5. 完整性验证

### 5.1 验证项

- 必需文件是否存在（state.json）
- JSON格式是否正确
- 必要字段是否完整（system_prompt, messages, timestamp）
- 系统Prompt哈希是否一致

### 5.2 监控指标

| 指标 | 告警阈值 |
|------|----------|
| 备份成功率 | < 95% |
| 恢复次数 | > 3/天 |
| 备份大小 | > 100MB |
| 恢复耗时 | > 30秒 |

---

## 6. 数据结构

### 6.1 快照格式

```json
{
  "version": "1.0",
  "timestamp": "2026-02-28_18-30-00",
  "session_id": "xxx",
  "system_prompt": "...",
  "system_prompt_hash": "sha256...",
  "messages": [...],
  "model": "minimax-m2.5",
  "message_count": 100,
  "metadata": {
    "total_tokens": 50000,
    "total_cost": 0.05
  }
}
```

---

*文档版本：1.0*
*最后更新：2026-02-28*
