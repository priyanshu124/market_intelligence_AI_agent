# Step 3 — Create and activate virtual environment

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

You'll see `(venv)` at the start of your terminal line. Every time you open this project, activate the venv first.

Tell VSCode to use this Python:
- Press `Ctrl+Shift+P`
- Type `Python: Select Interpreter`
- Choose the one that shows `./venv/...`

---

## Step 4 — Project structure

Current structure:

```text
market-intelligence_AI_agent/
│
├── venv/
├── data/
│   └── g2/
├── extract/
│   ├── g2/
│   ├── reddit/
│   └── src/
│       ├── g2/
│       │   ├── g2_config.py
│       │   ├── g2_scraper.py
│       │   ├── g2_build_ground_truth.py
│       │   ├── g2_debug_selectors.py
│       │   └── __init__.py
│       └── reddit/
│           └── __init__.py
├── main.py
├── test_api.py
├── .env
└── requirements.txt
```

Create folders (Windows PowerShell):

```powershell
New-Item -ItemType Directory -Force -Path data\g2, extract\g2, extract\reddit, extract\src\g2, extract\src\reddit
```

Install dependencies:

```powershell
pip install -r requirements.txt
playwright install chromium
```

Set up `.env`:

```env
# AI Config
AI_API_KEY=your_ai_api_key_here

# G2 Config
G2_USERNAME=your_g2_username
G2_PASSWORD=your_g2_password
API_KEY=your_api_key

# Reddit Config
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
REDDIT_USER_AGENT=ERP-Research-Bot/1.0
```