# 服务停止
`kill_service.sh` 只匹配当前仓库下 `open_jev_server` 和 `jev_omni_server`
的服务入口，并且只处理当前用户的进程。它不会按端口、Python 可执行文件或
用户名批量结束其他服务：

```bash
bash kill_service.sh --dry-run
bash kill_service.sh
```

Jev-Omni 也可以使用单 worker Gunicorn 部署。命令需要在
`jev_omni_server` 目录执行；`kill_service.sh` 会识别 Gunicorn master 和
worker，并通过 PID 文件对应的服务命令进行优雅停止：

```bash
cd ~/work/jev-decision-maker/jev_omni_server
conda activate omin-decision-maker
gunicorn --workers 1 --bind 0.0.0.0:8019 --daemon \
  --pid "$PWD/gunicorn.pid" \
  --access-logfile access.log --error-logfile error.log \
  -k uvicorn.workers.UvicornWorker server:app
```

使用 8019 时，请同步将 `.env` 中的 `JEV_OMNI_PORT` 和
`JEV_OMNI_PUBLIC_URL` 配成 8019。停止时仍只需执行仓库根目录的
`bash kill_service.sh`，不需要手动读取 PID 文件或按端口杀进程。

如果服务由 systemd 管理，请使用对应的 `systemctl stop` 单元；本脚本不会触碰
仓库目录之外的 systemd 服务。

# 环境
- conda activate jev-quantify
- conda activate omin-decision-maker
- conda activate open-decision-maker
