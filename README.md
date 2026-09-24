# 启动
- 你应该根据你要启动的服务、切换环境，然后再执行：
gunicorn --workers 3 --bind 0.0.0.0:8019 --daemon  --pid /home/ymtao/work/jev-decision-maker/gunicorn.pid --access-logfile access.log --error-logfile error.log -k uvicorn.workers.UvicornWorker xxxx.server:app

# 停止服务
- 同样，你应该根据自己已经启动的服务，选择要杀的服务，修改 APP_MARK="open_jev_server.server:app" 指向你的服务