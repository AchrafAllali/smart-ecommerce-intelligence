<div align="center">

# 🛒 Smart E-Commerce Intelligence Platform

**An end-to-end data platform that transforms 100K+ raw e-commerce transactions into business insights, ML predictions, and an AI-powered assistant.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

[Overview](#-overview) •
[Features](#-features) •
[Architecture](#-architecture) •
[Screenshots](#-screenshots) •
[Quick Start](#-quick-start) •
[Tech Stack](#-tech-stack) •
[Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Screenshots](#-screenshots)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Machine Learning Results](#-machine-learning-results)
- [API Endpoints](#-api-endpoints)
- [Testing](#-testing)
- [Development](#-development)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Overview

**Smart E-Commerce Intelligence Platform** is a full-stack data project built on the **Olist Brazilian E-Commerce dataset** (~100K orders, 2016–2018). It covers the entire data lifecycle:

```
Raw Data → ETL → SQL Analytics → ML Models → REST API → Web UI → AI Assistant
```

**What makes this project unique:**

- 🏗️ **Full data pipeline** — from raw CSVs to production-ready API
- 🤖 **Generative AI** — AI assistant powered by Groq (gpt-oss-120b) that answers business questions in natural language (FR/EN/AR)
- 🐳 **Dockerized** — one command to run everything
- ✅ **Tested** — 35 unit tests, 100% pass rate
- 📊 **MLOps-ready** — MLflow experiment tracking

---

## ✨ Features

| Module | Description |
|---|---|
| 📊 **Dashboard** | 8 interactive pages with KPIs, sales analytics, segments, churn, recommendations |
| 🏗️ **ETL Pipeline** | Extract from 9 CSV files, transform, load into SQLite star schema |
| 🔍 **SQL Analytics** | 16 production-ready queries for business KPIs |
| 📈 **EDA** | 9 automated visualizations (matplotlib + seaborn) |
| 👥 **Customer Segmentation** | RFM + K-Means → 5 business segments (VIP, Loyal, New, At Risk, Lost) |
| 🔮 **Churn Prediction** | XGBoost / Random Forest with **ROC-AUC 0.98** |
| 🛍️ **Recommendation System** | Hybrid CF (category + popularity) with **Hit Rate 88%** |
| 🌐 **REST API** | 11 endpoints (Flask) |
| 🤖 **AI Assistant** | Natural language → SQL → Answer (Groq + gpt-oss-120b, multilingual) |
| 🐳 **Docker** | Multi-container deployment (API + Webapp) |
| ✅ **Tests** | 35 pytest tests (API, ETL, ML, RAG, Webapp) |
| 📊 **MLflow** | Experiment tracking (params, metrics, models) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          RAW DATA                               │
│  9 Olist CSV files  →  customers, orders, products, payments…  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ETL PIPELINE                              │
│   Extract  →  Transform  →  Load (star schema)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SQLITE DATABASE                             │
│   fact_sales + dim_* + ML outputs (segments, churn, recos)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐         ┌──────────┐
   │   SQL   │         │    ML    │         │   RAG    │
   │Analytics│         │  Models  │         │  + LLM   │
   └────┬────┘         └────┬─────┘         └────┬─────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            ▼
              ┌──────────────────────────┐
              │     FLASK REST API       │
              │       (port 5000)        │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    FLASK WEB UI          │
              │       (port 5001)        │
              │  + AI Assistant chatbot  │
              └──────────────────────────┘
```

---

## 📸 Screenshots

### 🏠 Home Page
<img width="1920" height="957" alt="image" src="https://github.com/user-attachments/assets/bbf0e340-ed55-4cae-84f3-8bfe9e766cea" />

### 📊 Dashboard Overview
<img width="1920" height="962" alt="image" src="https://github.com/user-attachments/assets/51478eca-1f6c-499c-9c76-9dc6bfb72454" />

### 👥 Customer Segments
<img width="1920" height="959" alt="image" src="https://github.com/user-attachments/assets/3370cb7c-a4ec-4c00-bbb2-62c350dbad7d" />

### 🔮 Churn Prediction
<img width="1920" height="961" alt="image" src="https://github.com/user-attachments/assets/1f1eb7f9-520c-47f2-84c5-286f3569afc9" />

### 🤖 AI Assistant
<img width="1920" height="959" alt="image" src="https://github.com/user-attachments/assets/71a4969b-39c6-474e-8a95-57a52c5d2aa3" />

### 📊 MLflow Tracking
<img width="1920" height="958" alt="image" src="https://github.com/user-attachments/assets/2b14dd68-6583-48bb-8b58-e22c1f1e0e76" />


> 💡 Add your own screenshots in `docs/screenshots/`

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker Desktop** (optional, for containerized deployment)
- **Groq API key** (free at [console.groq.com](https://console.groq.com/keys))

### Option 1 — Docker (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/smart-ecommerce-intelligence.git
cd smart-ecommerce-intelligence

# 2. Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY

# 3. Launch everything
docker-compose up --build
```

**Access:**
- 🌐 Web UI → http://localhost:5001
- 🔌 REST API → http://localhost:5000
- 📊 MLflow UI → http://localhost:5050

### Option 2 — Local (Python)

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/smart-ecommerce-intelligence.git
cd smart-ecommerce-intelligence
pip install -r requirements.txt

# 2. Place Olist CSVs in data/raw/

# 3. Run the ETL pipeline
python main.py

# 4. Train ML models
python -m ml.segmentation
python -m ml.churn
python -m ml.recommendation

# 5. Launch the webapp
python -m webapp.app
```

Access at http://localhost:5001

---

## 📁 Project Structure

```
smart-ecommerce-intelligence/
│
├── api/                          # REST API (Flask)
│   └── app.py                    # 11 endpoints
│
├── webapp/                       # Web UI (Flask + Jinja2)
│   ├── app.py                    # Routes
│   ├── templates/                # 8 HTML pages
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── index.html
│   │   ├── sales.html
│   │   ├── segments.html
│   │   ├── churn.html
│   │   ├── recommendations.html
│   │   ├── assistant.html
│   │   └── explorer.html
│   └── static/
│       └── style.css             # Custom CSS (gradients, icons)
│
├── etl/                          # Data pipeline
│   ├── extract.py
│   ├── transform.py
│   └── load.py
│
├── analytics/                    # Data analysis
│   ├── queries.sql               # 16 SQL queries
│   ├── run_queries.py
│   └── eda.py                    # 9 visualizations
│
├── ml/                           # Machine Learning
│   ├── segmentation.py           # RFM + K-Means
│   ├── churn.py                  # XGBoost / RandomForest
│   ├── recommendation.py         # Hybrid CF
│   ├── mlflow_tracking.py        # MLflow experiments
│   └── models/                   # Serialized models
│
├── rag/                          # GenAI / LLM
│   ├── sql_agent.py              # NL → SQL
│   ├── assistant.py              # CLI chatbot
│   ├── config.py
│   └── prompts.py
│
├── tests/                        # 35 pytest tests
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_etl.py
│   ├── test_ml.py
│   ├── test_rag.py
│   └── test_webapp.py
│
├── database/                     # SQLite database
│   └── ecommerce.db
│
├── dashboard/                    # Generated PNGs (EDA + ML)
│
├── data/                         # Raw + processed data
│   ├── raw/
│   └── processed/
│
├── notebooks/                    # (Optional) Jupyter notebooks
│
├── docs/                         # Documentation + screenshots
│   └── screenshots/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

<table>
<tr>
<td><b>Category</b></td>
<td><b>Technologies</b></td>
</tr>
<tr>
<td><b>Languages</b></td>
<td>Python 3.10, SQL, HTML, CSS, JavaScript</td>
</tr>
<tr>
<td><b>Data Engineering</b></td>
<td>Pandas, NumPy, SQLite, ETL, Star Schema</td>
</tr>
<tr>
<td><b>Data Analysis</b></td>
<td>SQL (CTEs, Window Functions), Matplotlib, Seaborn, Plotly.js</td>
</tr>
<tr>
<td><b>Machine Learning</b></td>
<td>scikit-learn, XGBoost, K-Means, Random Forest, Logistic Regression</td>
</tr>
<tr>
<td><b>GenAI / LLM</b></td>
<td>Groq, gpt-oss-120b, LangChain, NL→SQL</td>
</tr>
<tr>
<td><b>Backend</b></td>
<td>Flask, Flask-CORS, Jinja2, REST API</td>
</tr>
<tr>
<td><b>Frontend</b></td>
<td>HTML5, CSS3 (gradients), JavaScript, Plotly.js, Font Awesome</td>
</tr>
<tr>
<td><b>DevOps</b></td>
<td>Docker, docker-compose, WSL 2</td>
</tr>
<tr>
<td><b>MLOps</b></td>
<td>MLflow (experiment tracking, model registry)</td>
</tr>
<tr>
<td><b>Testing</b></td>
<td>pytest, pytest-cov, pytest-flask</td>
</tr>
</table>

---

## 📊 Machine Learning Results

### 🎯 Customer Segmentation (RFM + K-Means)

| Segment | Customers | % | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|---|
| **VIP** | 18,461 | 19.4% | 137 days | 1.13 | 245.30 BRL |
| **Loyal** | 20,748 | 21.7% | 109 days | 1.00 | 179.23 BRL |
| **New** | 20,557 | 21.5% | 271 days | 1.01 | 61.13 BRL |
| **At Risk** | 20,174 | 21.1% | 383 days | 1.03 | 268.43 BRL |
| **Lost** | 15,480 | 16.2% | 332 days | 1.00 | 59.71 BRL |

### 🔮 Churn Prediction

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** |
|---|---|---|---|---|---|
| Logistic Regression | 0.8953 | 1.0000 | 0.8269 | 0.9052 | 0.9695 |
| Random Forest | 0.9014 | 0.9948 | 0.8414 | 0.9117 | **0.9802** ⭐ |
| **XGBoost** | **0.9029** | 0.9645 | **0.8715** | **0.9157** | 0.9757 |

### 🛍️ Recommendation System

- **Method**: Hybrid (Category-based + Popularity)
- **Hit Rate@10**: **88.00%** (random baseline: 1.39%)
- **Coverage**: 95,420 customers → 477,100 recommendations

### 📈 Business KPIs

| Metric | Value |
|---|---|
| Total Revenue | **15,843,553.24 BRL** |
| Total Orders | **98,666** |
| Total Customers | **95,420** |
| Average Order Value | **160.58 BRL** |
| Total Products | **32,951** |
| Time Range | Sep 2016 – Oct 2018 |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/kpis` | Global KPIs |
| `GET` | `/api/monthly-sales` | Monthly revenue + orders |
| `GET` | `/api/top-categories?limit=N` | Top categories |
| `GET` | `/api/top-cities?limit=N` | Top cities |
| `GET` | `/api/segments` | Segment summary |
| `GET` | `/api/segment/<customer_id>` | Segment of a customer |
| `GET` | `/api/churn/<customer_id>` | Churn probability |
| `GET` | `/api/churn/top-risk?limit=N` | Top churn-risk customers |
| `GET` | `/api/recommendations/<customer_id>` | Product recommendations |
| `GET` | `/api/popular-products?limit=N` | Top popular products |
| `POST` | `/api/assistant` | AI chatbot (NL → SQL → answer) |

### Example

```bash
curl http://localhost:5000/api/kpis
```

```json
{
  "total_revenue": 15843553.24,
  "total_orders": 98666,
  "total_customers": 95420,
  "avg_order_value": 160.58
}
```

---

## ✅ Testing

The project has **35 unit tests** with **100% pass rate**.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_api.py -v

# Run one test
pytest tests/test_api.py::TestRESTAPI::test_kpis -v
```

**Coverage summary:**

| Module | Coverage |
|---|---|
| `api/app.py` | 82% |
| `webapp/app.py` | 76% |
| `tests/test_etl.py` | 100% |
| `tests/test_ml.py` | 100% |
| `tests/test_webapp.py` | 95% |

---

## 💻 Development

### Setup the dev environment

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run MLflow UI

```bash
mlflow ui --port 5050
# Open http://localhost:5050
```

### Run the AI Assistant (CLI)

```bash
python -m rag.assistant
```

Ask questions like:
- 🇫🇷 "Quel est le chiffre d'affaires total ?"
- 🇬🇧 "Top 5 categories by revenue"
- 🇸🇦 "ما هو إجمالي الإيرادات؟"

### Environment variables

Create a `.env` file:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Achraf Allali**

- 🐙 GitHub: [https://github.com/AchrafAllali](https://github.com/AchrafAllali)
- 💼 LinkedIn: [https://www.linkedin.com/in/achraf-allali-9889a0321/](https://www.linkedin.com/in/achraf-allali-9889a0321/)
- 📧 Email: achrafallali2003@gmail.com

---

## ❤️ Acknowledgments

- Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- LLM: [Groq](https://groq.com/) with `gpt-oss-120b`
- Icons: [Font Awesome](https://fontawesome.com/)
- Charts: [Plotly.js](https://plotly.com/javascript/)

---

<div align="center">

**⭐ If you found this project useful, please give it a star! ⭐**

Made with ❤️ and Python

</div>
