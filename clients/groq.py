from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_completion(messages, tools_schema=None):
    try:
        params = {
            "model": "llama-3.3",
            "messages": messages,
        }
        if tools_schema:
            params["tools"] = [{"type": "function", "function": t} for t in tools_schema]

        response = client.chat.completions.create(**params)
        message = response.choices[0].message

        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        }
    except Exception as e:
        return {"role": "assistant", "content": f"Groq Error: {str(e)}", "tool_calls": None}
