# ChatTool

一个针对 `ghc-api` 的轻量聊天前端。它会读取代理的模型列表，并按模型的 `supported_endpoints` 自动选择：

- `/v1/responses`
- `/v1/chat/completions`
- `/v1/messages`

界面只保留必要能力：

- 模型选择
- 多会话管理
- 浏览器本地保存会话

## 当前行为

- 模型列表来自 `http://127.0.0.1:8313/v1/models/full/`
- `gpt-5` / `o*` 类模型优先走 `/v1/responses`
- `claude*` 且支持 `/v1/messages` 的模型优先走 `/v1/messages`
- 其他模型默认走 `/v1/chat/completions`
- 不再暴露 `provider`、`temperature`、`system prompt` 这些 UI 参数

## Windows 本地测试

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 13579 --reload
```

访问：

- `http://127.0.0.1:13579`

前提：

- `ghc-api` 已经启动在 `http://127.0.0.1:8313`
- `http://127.0.0.1:8313/v1/models/full/` 可访问

## Linux 单机部署

```bash
cd /opt/chattool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 13579
```

可选：配成 systemd，参考 `deploy/chattool.service.example`。

## 多会话

- 会话保存在浏览器 `localStorage`
- 不写数据库
- 换浏览器或清空站点数据后会丢失
