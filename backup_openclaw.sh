#!/bin/bash
# OpenClaw 备份脚本
# 用法: ./backup_openclaw.sh

REPO="https://ghp_你的PAT@github.com/lichaonetuser/openclaw_backup.git"
BACKUP_DIR="$HOME/.openclaw/backup-git"

echo "=== OpenClaw 备份开始 ==="

# 创建备份目录（如果不存在）
mkdir -p "$BACKUP_DIR"

# 复制配置文件
cp -r "$HOME/.openclaw/openclaw.json" "$BACKUP_DIR/"
cp -r "$HOME/.openclaw/agents" "$BACKUP_DIR/"
cp -r "$HOME/.openclaw/memory" "$BACKUP_DIR/"

# 进入备份目录
cd "$BACKUP_DIR"

# 初始化 git（如果需要）
if [ ! -d ".git" ]; then
    git init
    git remote add origin "$REPO"
fi

# 添加所有文件
git add -A

# 提交
git commit -m "OpenClaw配置备份 - $(date '+%Y-%m-%d %H:%M')" || echo "没有新文件需要提交"

# 推送
git push origin main

echo "=== 备份完成 ==="