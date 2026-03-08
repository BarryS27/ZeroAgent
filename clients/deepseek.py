import openai
import os

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=DEEPSEEK_BASE_URL
)

def get_completion(messages, tools_schema=None):
    try:
        params = {
            "model": "deepseek-chat",
            "messages": messages,
        }

        if tools_schema:
            params["tools"] = [{"type": "function", "function": t} for t in tools_schema]
            params["tool_choice"] = "auto"

        response = client.chat.completions.create(**params)

        assistant_message = response.choices[0].message

        return {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls
        }

    except Exception as e:
        return {"role": "assistant", "content": f"DeepSeek Error: {str(e)}", "tool_calls": None}
