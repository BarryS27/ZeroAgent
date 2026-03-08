import google.generativeai as genai
import os
import json

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_completion(messages, tools_schema=None):
    try:
        model_tools = None
        if tools_schema:
            model_tools = [{"function_declaration": t} for t in tools_schema]

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=model_tools
        )

        gemini_history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if msg["role"] != "system":
                gemini_history.append({"role": role, "parts": [msg["content"]]})

        last_message = gemini_history[-1]["parts"][0]
        history_except_last = gemini_history[:-1]

        chat = model.start_chat(history=history_except_last)
        response = chat.send_message(last_message)

        content_text = ""
        tool_calls = []

        for part in response.candidates[0].content.parts:
            if part.text:
                content_text += part.text

            if part.function_call:
                class MockToolCall:
                    def __init__(self, fc):
                        self.id = "gen-id"
                        self.function = type('obj', (object,), {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args)
                        })
                tool_calls.append(MockToolCall(part.function_call))

        return {
            "role": "assistant",
            "content": content_text if content_text else None,
            "tool_calls": tool_calls if tool_calls else None
        }

    except Exception as e:
        return {"role": "assistant", "content": f"Gemini Error: {str(e)}", "tool_calls": None}
