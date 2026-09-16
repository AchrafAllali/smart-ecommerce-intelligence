"""
Test Groq connection + simple chat.
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

print("=" * 60)
print("GROQ TEST")
print("=" * 60)

if not api_key:
    print("❌ GROQ_API_KEY not set")
    exit(1)

print(f"✓ API key: {api_key[:12]}...{api_key[-6:]}")
print(f"✓ Model: {model}\n")

client = Groq(api_key=api_key)

# List models
models = client.models.list()
available = sorted([m.id for m in models.data])

print(f"✅ {len(available)} models available:")
for m in available:
    marker = " ← YOUR CHOICE" if m == model else ""
    print(f"   • {m}{marker}")

if model not in available:
    print(f"\n❌ Model '{model}' not available!")
    exit(1)

print(f"\n✅ Model '{model}' is AVAILABLE\n")

# Test chat
print("=" * 60)
print("CHAT TEST")
print("=" * 60)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'Hello!' in one word."}],
        temperature=0,
    )
    print(f"✓ Response: {response.choices[0].message.content}")
    print("\n🎉 Groq is fully working!")
except Exception as e:
    print(f"❌ Error: {e}")