import anthropic
import os

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_completion(messages, tools_schema=None):
    try:
        system_prompt = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)

        anthropic_tools = []
        if tools_schema:
            for t in tools_schema:
                anthropic_tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"]
                })

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            system=system_prompt,
            messages=filtered_messages,
            tools=anthropic_tools if anthropic_tools else None
        )

        content_text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                class MockToolCall:
                    def __init__(self, b):
                        self.id = b.id
                        self.function = type('obj', (object,), {
                            "name": b.name,
                            "arguments": str(b.input).replace("'", '"')
                        })
                tool_calls.append(MockToolCall(block))

        return {
            "role": "assistant",
            "content": content_text if content_text else None,
            "tool_calls": tool_calls if tool_calls else None
        }

    except Exception as e:
        return {"role": "assistant", "content": f"Anthropic Error: {str(e)}", "tool_calls": None}
