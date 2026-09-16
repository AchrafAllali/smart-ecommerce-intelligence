"""
CLI Chatbot: interact with the e-commerce database in natural language.

Usage:
    python -m rag.assistant
"""

import sys
from rag.sql_agent import ask_safe


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🤖  Smart E-Commerce AI Assistant                          ║
║   ─────────────────────────────────────────                  ║
║   Ask questions about your e-commerce data in                ║
║   French, English, or Arabic.                                ║
║                                                              ║
║   Type 'exit' or 'quit' to leave.                            ║
║   Type 'examples' to see sample questions.                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAMPLES = """
📌 EXEMPLE DE QUESTIONS / SAMPLE QUESTIONS / أسئلة نموذجية

FR:
- Quel est le chiffre d'affaires total ?
- Quelles sont les 5 meilleures catégories ?
- Combien de clients ont churné ?
- Quelle est la ville avec le plus de ventes ?
- Quel est le panier moyen ?

EN:
- What is the total revenue?
- Top 5 categories by revenue
- How many customers churned?
- Which city has the most sales?
- What is the average order value?

AR:
- ما هو إجمالي الإيرادات؟
- ما هي أفضل 5 فئات؟
- كم عدد العملاء الذين غادروا؟
"""


def main():
    print(BANNER)

    while True:
        try:
            question = input("\n💬 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("👋 Bye!")
            break

        if question.lower() == "examples":
            print(EXAMPLES)
            continue

        print("\n🤔 Thinking...")
        result = ask_safe(question)

        if result["success"]:
            print(f"\n🤖 Assistant:\n{result['answer']}")
        else:
            print(f"\n❌ {result['answer']}")


if __name__ == "__main__":
    main()