# Jaipur AI Agent Setup Guide

## Initial Setup

### 1. Clone the Repository
```bash
git clone https://github.com/danielh-hong/rob311-Project.git
cd to root
```

### 2. Create Virtual Environment
```bash
# Create the virtual environment
python3 -m venv venv
```

### 3. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt when activated.

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist yet, install manually:
```bash
pip install bazaar-ai
```

Then create the requirements file:
```bash
pip freeze > requirements.txt
```

## Running the Simulation
```bash
bazaar-simulate
```

This will start a server and open a web browser where you can test agents.

## Working on the Project

### Every time you start working:

1. Open terminal and navigate to project directory
2. Activate the virtual environment:
```bash
   source venv/bin/activate  # macOS/Linux
   # OR
   venv\Scripts\activate     # Windows
```
3. Start coding!

### When you're done:
```bash
deactivate
```

## Troubleshooting

### "python3: command not found"
Try `python --version` instead. Use `python` wherever you see `python3`.

### "pip: command not found" 
Your Python installation may not include pip. Download get-pip.py from:
https://bootstrap.pypa.io/get-pip.py

Then run:
```bash
python get-pip.py
```

## Notes

- The `venv/` folder should **never** be committed to git (it's in `.gitignore`)
- Always activate the virtual environment before working on the project
- If you install new packages, update requirements.txt: `pip freeze > requirements.txt`