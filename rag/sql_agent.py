"""
Robust NL → SQL pipeline with rule-based intent routing + deterministic table formatting.

Strategy:
    1. Detect intent (smalltalk vs data) via LLM.
    2. For DATA: generate SQL via LLM, execute, then:
        - If result is a single value → let LLM phrase it naturally.
        - If result is a TABLE (multi-rows) → format it deterministically in Python.
"""

import re
import sqlite3
import warnings

import numpy as np
import pandas as pd

from rag.config import (
    DB_PATH,
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    TEMPERATURE,
)

warnings.filterwarnings("ignore")


# ============================================================
# LLM INIT (singleton)
# ============================================================
_llm = None


def get_llm():
    """Return the best available LLM (Groq → OpenAI → Ollama)."""
    global _llm
    if _llm is not None:
        return _llm

    # Priority 1: Groq (fastest + free)
    if LLM_PROVIDER in ("groq", "auto") and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        print(f"[INFO] LLM: Groq ({GROQ_MODEL})")
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            api_key=GROQ_API_KEY,
        )
        return _llm

    # Priority 2: OpenAI
    if LLM_PROVIDER in ("openai", "auto") and OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        print(f"[INFO] LLM: OpenAI ({OPENAI_MODEL})")
        _llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
        return _llm

    # Priority 3: Ollama (local fallback)
    from langchain_ollama import ChatOllama
    print(f"[INFO] LLM: Ollama ({OLLAMA_MODEL}) @ {OLLAMA_BASE_URL}")
    _llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
    )
    return _llm


def llm():
    return get_llm()


# ============================================================
# DATABASE
# ============================================================
def execute_sql(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


# ============================================================
# INTENT CLASSIFIER (single LLM call)
# ============================================================
INTENT_PROMPT = """You are an intent classifier for an e-commerce data assistant.

Classify the user's message into EXACTLY ONE of these categories:

- SMALLTALK  : greetings, thanks, "how are you", "who are you", "help", "bye" — any conversational message NOT asking about data
- DATA       : any question about the e-commerce data (revenue, orders, customers, products, categories, cities, churn, recommendations, payments, reviews, delivery, segments, KPIs)

Examples:
"سلام" → SMALLTALK
"bonjour" → SMALLTALK
"comment allez-vous ?" → SMALLTALK
"qui es-tu ?" → SMALLTALK
"merci" → SMALLTALK
"aide" → SMALLTALK
"Quel est le chiffre d'affaires total ?" → DATA
"Top 5 catégories par revenu" → DATA
"Combien de clients ont churné ?" → DATA
"ما هي إجمالي الإيرادات؟" → DATA

Reply with ONLY one word: SMALLTALK or DATA.

Message: {question}
Category:"""


def classify_intent(question: str) -> str:
    """Returns 'smalltalk' or 'data'."""
    response = llm().invoke(INTENT_PROMPT.format(question=question))
    text = response.content.strip().upper()
    if "SMALLTALK" in text:
        return "smalltalk"
    return "data"


# ============================================================
# SMALL TALK RESPONSE (LLM-powered)
# ============================================================
SMALLTALK_PROMPT = """You are a friendly AI assistant for a Smart E-Commerce Intelligence Platform (Olist dataset).

Respond to the user's message in the SAME LANGUAGE they used.
Be warm, brief (1-2 sentences), and helpful.

Rules:
- If they greet you → greet back and mention you can answer questions about e-commerce data.
- If they thank you → say "you're welcome" warmly.
- If they ask who you are → explain you analyze e-commerce data (sales, customers, churn, recommendations).
- If they ask for help → suggest 3-4 example questions.
- NEVER invent data. NEVER mention numbers unless the user asked for data.
- Answer in French, English, or Arabic (match the user's language).

User: {question}
Assistant:"""


def handle_smalltalk(question: str) -> str:
    response = llm().invoke(SMALLTALK_PROMPT.format(question=question))
    return response.content.strip()


# ============================================================
# NL → SQL
# ============================================================
from rag.prompts import SCHEMA_DESCRIPTION

SQL_PROMPT = """You are an expert SQLite analyst for an e-commerce database.

DATABASE SCHEMA:
{schema}

TASK:
Convert the user's question into a SINGLE valid SQLite SELECT query.

STRICT RULES:
- Return ONLY the SQL query. No markdown, no ```sql, no explanation.
- Use ONLY SELECT statements.
- Use EXACT table and column names from the schema.
- The currency is Brazilian Real (BRL).
- For monetary values, use ROUND(SUM(...), 2).
- Always end with a semicolon.

EXAMPLES:
Q: Quel est le chiffre d'affaires total ?
A: SELECT ROUND(SUM(total_amount), 2) AS total_revenue FROM fact_sales;

Q: Top 5 catégories par revenu
A: SELECT product_category_name_english AS category, ROUND(SUM(total_amount), 2) AS revenue FROM fact_sales GROUP BY product_category_name_english ORDER BY revenue DESC LIMIT 5;

Q: Combien de clients ont churné ?
A: SELECT SUM(churn) AS total_churned FROM customer_churn;

Q: Quelle ville a le plus de ventes ?
A: SELECT customer_city, ROUND(SUM(total_amount), 2) AS revenue FROM fact_sales GROUP BY customer_city ORDER BY revenue DESC LIMIT 1;

Q: Quel est le panier moyen ?
A: SELECT ROUND(SUM(total_amount) / COUNT(DISTINCT order_id), 2) AS avg_order_value FROM fact_sales;

QUESTION: {question}

SQL:"""


def generate_sql(question: str) -> str:
    response = llm().invoke(
        SQL_PROMPT.format(schema=SCHEMA_DESCRIPTION, question=question)
    )
    text = response.content.strip()
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"^SQL:\s*", "", text, flags=re.IGNORECASE)
    if ";" in text:
        text = text.split(";")[0] + ";"
    return text.strip()


# ============================================================
# SQL RESULT → NATURAL LANGUAGE (single value only)
# ============================================================
ANSWER_PROMPT = """You are a senior data analyst. Answer the user's question using ONLY the SQL result below.

STRICT RULES:
- Answer in the SAME LANGUAGE as the question (French, English, or Arabic).
- Be concise (1-2 sentences max).
- Currency is ALWAYS "BRL" (Brazilian Real), NEVER "$" or "€" or "USD".
- Format numbers with thousands separators.
- Do NOT invent facts. Only use numbers present in the result.

QUESTION:
{question}

SQL:
{sql}

RESULT:
{result}

ANSWER (1-2 sentences):"""


# ============================================================
# DETERMINISTIC TABLE FORMATTER (Python, not LLM)
# ============================================================
def _format_table_answer(question: str, df: pd.DataFrame) -> str:
    """Format a table result deterministically (no LLM hallucination)."""
    n_rows = len(df)
    q_lower = question.lower()

    # Detect the requested "top N" from the question
    requested_n = None
    match = re.search(r"\b(?:top|meilleurs?|premiers?|أفضل)\s*(\d+)", q_lower)
    if match:
        requested_n = int(match.group(1))
    else:
        # Look for any standalone number
        match = re.search(r"\b(\d+)\b", q_lower)
        if match:
            n = int(match.group(1))
            if 1 <= n <= 100:
                requested_n = n

    n_to_show = requested_n if requested_n else n_rows
    n_to_show = min(n_to_show, n_rows)

    # Build the header based on question keywords
    if any(k in q_lower for k in ["catégorie", "categorie", "category", "فئة", "فئات"]):
        header = f"🏆 **Top {n_to_show} catégories par chiffre d'affaires :**"
    elif any(k in q_lower for k in ["ville", "villes", "city", "cities", "مدينة", "مدن"]):
        header = f"🌆 **Top {n_to_show} villes par chiffre d'affaires :**"
    elif any(k in q_lower for k in ["produit", "produits", "product", "منتج", "منتجات"]):
        header = f"📦 **Top {n_to_show} produits les plus vendus :**"
    elif any(k in q_lower for k in ["client", "clients", "customer", "عميل", "عملاء"]):
        header = f"💎 **Top {n_to_show} meilleurs clients :**"
    elif any(k in q_lower for k in ["paiement", "payment", "دفع"]):
        header = f"💳 **Méthodes de paiement :**"
    elif any(k in q_lower for k in ["avis", "review", "note", "تقييم"]):
        header = f"⭐ **Distribution des avis :**"
    elif any(k in q_lower for k in ["segment", "segments"]):
        header = f"👥 **Segments clients :**"
    elif any(k in q_lower for k in ["mois", "month", "mensuel", "شهر"]):
        header = f"📅 **Évolution mensuelle :**"
    else:
        header = f"📊 **Résultats ({n_to_show} lignes) :**"

    lines = [header, ""]
    cols = list(df.columns)

    for i, (_, row) in enumerate(df.head(n_to_show).iterrows(), 1):
        label = str(row[cols[0]])

        # Build values part
        value_parts = []
        for col in cols[1:]:
            val = row[col]
            col_lower = col.lower()

            # Format numbers
            if isinstance(val, (int, np.integer)):
                formatted = f"{int(val):,}".replace(",", " ")
            elif isinstance(val, (float, np.floating)):
                if any(k in col_lower for k in ["revenue", "amount", "value", "spent", "monetary", "price", "total"]):
                    formatted = f"{val:,.2f} BRL".replace(",", " ")
                elif any(k in col_lower for k in ["pct", "percent", "percentage", "rate"]):
                    formatted = f"{val:.2f}%"
                else:
                    formatted = f"{val:,.2f}".replace(",", " ")
            else:
                formatted = str(val)

            value_parts.append(f"**{formatted}**")

        values = " — ".join(value_parts)
        lines.append(f"{i}. **{label}** — {values}")

    if n_rows > n_to_show:
        lines.append(f"\n_(... et {n_rows - n_to_show} autres lignes)_")

    return "\n".join(lines)


# ============================================================
# ANSWER GENERATION
# ============================================================
def generate_answer(question: str, sql: str, df: pd.DataFrame) -> str:
    """
    Smart answer generation:
    - Table (multi-rows) → format directly in Python.
    - Single value       → use LLM for a natural sentence.
    """
    if df.empty:
        return "Aucun résultat trouvé pour cette requête."

    print(f"[DEBUG] Rows in result: {len(df)}")

    # Case 1: Table with multiple rows → format in Python
    if len(df) >= 2 and len(df.columns) >= 2:
        return _format_table_answer(question, df)

    # Case 2: Single value (or single row) → ask the LLM
    preview = df.to_string(index=False)
    prompt = ANSWER_PROMPT.format(question=question, sql=sql, result=preview)
    response = llm().invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()


# ============================================================
# MAIN API
# ============================================================
def ask(question: str) -> str:
    """Full pipeline."""
    # Step 1: Intent detection
    intent = classify_intent(question)
    print(f"[DEBUG] Intent: {intent}")

    if intent == "smalltalk":
        return handle_smalltalk(question)

    # Step 2: Generate SQL
    sql = generate_sql(question)
    print(f"[DEBUG] SQL: {sql}")

    # Step 3: Execute
    try:
        df = execute_sql(sql)
    except Exception as e:
        return f"❌ Erreur SQL : {e}\n\nSQL généré :\n{sql}"

    # Step 4: Answer (table in Python, single value via LLM)
    return generate_answer(question, sql, df)


def ask_safe(question: str) -> dict:
    try:
        return {"success": True, "answer": ask(question)}
    except Exception as e:
        return {"success": False, "answer": f"Error: {e}"}