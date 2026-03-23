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

### 5. Clone Bazaar-AI Source
```bash
git clone https://github.com/chandra-gummaluru/bazaar-ai.git bazaar-ai
```

## Running the Simulation (Web UI)
```bash
cd bazaar-ai
.\launch-bazaar-ai.bat    # Windows
bash launch-bazaar-ai     # macOS/Linux
```

This will start a local web server and open your browser to the UI where you can:
- Select your agent from the dropdown
- Choose an opponent (SmartAgent, RandomAgent, etc.)
- Watch them compete live

## Working on the Project

### Every time you start working:

1. Open terminal and navigate to project directory:
```bash
cd C:\Users\danie\Documents\GitHub\rob311-Project  # Windows
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

3. To test your agent visually:
```bash
cd bazaar-ai
.\launch-bazaar-ai.bat    # Windows
bash launch-bazaar-ai      # macOS/Linux
```

4. To edit your agent:
- Edit `bazaar-ai/agents/custom_agent.py`
- Modify the `genome` parameters or scoring logic
- Reload the web UI to test changes

### When you're done:
```bash
deactivate
```

## Submission

Your agent file is located at:
```
bazaar-ai/agents/custom_agent.py
```

**Submit this single file to AutoLab by the deadline.**

## More Commands

For a complete list of all commands, see [COMMANDS.md](COMMANDS.md).

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
- The `bazaar-ai/` folder contains the game framework source (committed as plain files, no nested git)
- Your submission agent is `bazaar-ai/agents/custom_agent.py`
- Always activate the virtual environment before working on the project
- If you make significant changes to your agent strategy, test it via the web UI first