# final_project

## Setup

This project is recommended to run on Python 3.11 for full compatibility.

### 1. Create and activate virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run API server

```powershell
python -m api.app
```

Then open:

- http://127.0.0.1:5000

## Notes

- Python 3.11 is the recommended baseline for this repository.
- Python 3.12+ may fail on some optional branches (for example PaddleOCR
  or legacy TruFor ecosystem dependencies).
