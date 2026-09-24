# 服务停止
`kill_service.sh` 只匹配当前仓库下 `open_jev_server` 和 `jev_omni_server`
的服务入口，并且只处理当前用户的进程。它不会按端口、Python 可执行文件或
用户名批量结束其他服务：

```bash
bash kill_service.sh --dry-run
bash kill_service.sh
```

如果服务由 systemd 管理，请使用对应的 `systemctl stop` 单元；本脚本不会触碰
仓库目录之外的 systemd 服务。

# 环境
- conda activate jev-quantify
- conda activate omin-decision-maker
- conda activate open-decision-maker
