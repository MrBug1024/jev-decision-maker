# Jev-Omni MCP

The service exposes one standard MCP Streamable HTTP endpoint:

```text
http://<server>:8020/mcp
```

Create an account in the web console, create an access Key, and send it as:

```text
Authorization: Bearer jev_live_...
```

The public tool keeps the same shape as the original gateway:

```json
{
  "situation": "The meeting starts at 10 AM. It is now 9 AM.",
  "questions": [
    {
      "type": "choice",
      "question": "What should happen next?",
      "options": ["Wait", "Start now"]
    },
    {
      "type": "yes_no",
      "question": "Has the meeting started?"
    }
  ]
}
```

The server returns one result per question in input order. `choice` returns a selected option and probabilities, `score` returns a normalized ordered score and probabilities, and `yes_no` returns the probability of `Yes` in `noul` plus the underlying distribution.

The browser console additionally supports `text`, `image`, `audio`, and `video` test inputs. MCP requests remain text-only until a media transport is added to the public tool schema; this keeps the existing client contract stable.
