# MCP 接入

JEV Gateway 只有一个服务进程和一个标准 MCP Streamable HTTP 端点：

```text
http://<服务地址>:8019/mcp
```

在网页控制台注册账户并创建 Key 后，将 Key 作为 Bearer Token 发送：

```text
Authorization: Bearer jev_live_...
```

可用工具为 `jev_decide`，输入结构如下：

```json
{
  "situation": "需要判断的背景文本",
  "questions": [
    {
      "type": "choice",
      "question": "应该采取哪种行动？",
      "options": ["选项 A", "选项 B"]
    },
    {
      "type": "score",
      "question": "这件事有多紧急？",
      "options": ["低", "中", "高"]
    },
    {
      "type": "yes_no",
      "question": "这条陈述是否成立？"
    }
  ]
}
```

返回 `results` 会与输入问题保持相同顺序。`choice` 返回选项概率，`score` 返回连续分数和概率，`yes_no` 返回陈述为真的概率。

完整启动、配置和安全说明请查看 [USAGE.md](USAGE.md)。
