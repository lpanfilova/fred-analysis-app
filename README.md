# FRED Analysis App

An AI chat application for exploring U.S. GDP and Core CPI data from FRED.
Ask questions, analyze historical trends, compare growth and inflation, and generate charts.

## Start and use

Requires macOS, Python 3.14, an OpenAI API key, and a FRED MCP Connector account.

From the cloned repository:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPEN_AI_API_KEY` in `.env`, then start the web app:

```sh
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501. Ask a question or choose a suggested prompt, then use
follow-up questions to explore further. Charts appear in the conversation and can be downloaded.

On first use, complete FRED sign-in in the browser and allow Keychain access if prompted.
Credentials are stored in macOS Keychain.

For terminal chat:

```sh
.venv/bin/python mcp_client.py --chat
```

## Test

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```