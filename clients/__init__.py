import importlib

PROVIDERS = [
    "openai",
    "claude",
    "gemini",
    "deepseek",
    "groq"
]

CLIENT_MAP = {}

for name in PROVIDERS:
    try:
        module = importlib.import_module(f".{name}", package=__name__)

        CLIENT_MAP[name] = getattr(module, "get_completion")
    except (ImportError, AttributeError):
        CLIENT_MAP[name] = None

def get_client(name):
    name = name.lower()
    if name not in CLIENT_MAP:
        available = ", ".join([k for k, v in CLIENT_MAP.items() if CLIENT_MAP[k] is not None])
        raise ValueError(f"❌ Unknown client: '{name}'. Current available: {available}")

    func = CLIENT_MAP[name]
    if func is None:
        raise ImportError(f"❌ Module '{name}.py' failed. Please check the file.")

    return func
