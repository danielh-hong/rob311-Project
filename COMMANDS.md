# Bazaar-AI Project Commands

## One-Time Setup

### 1. Clone Bazaar-AI (if not already done)
```powershell
cd C:\Users\danie\Documents\GitHub\rob311-Project
git clone https://github.com/chandra-gummaluru/bazaar-ai.git .\bazaar-ai
```

If folder already exists:
```powershell
cd C:\Users\danie\Documents\GitHub\rob311-Project\bazaar-ai
git pull
```

## Every Time You Work

### Activate Virtual Environment
```powershell
cd C:\Users\danie\Documents\GitHub\rob311-Project
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

### Launch the Web UI (to test visually)
```powershell
cd C:\Users\danie\Documents\GitHub\rob311-Project\bazaar-ai
.\launch-bazaar-ai.bat
```

Browser opens to `http://localhost:5000` automatically.

**To test your agent:**
1. Select `CustomAgent` from first dropdown
2. Select `SmartAgent` or `RandomAgent` from second dropdown
3. Click **Run Game**

---

## Submission

### Your Agent File Location
```
C:\Users\danie\Documents\GitHub\rob311-Project\bazaar-ai\agents\custom_agent.py
```

**On AutoLab submission deadline:**
- Upload `custom_agent.py` as your single agent file
- It will be tested against the grading reference agents automatically

---

## Working on the Project

### View Current Agent Code
```powershell
code C:\Users\danie\Documents\GitHub\rob311-Project\bazaar-ai\agents\custom_agent.py
```

### Edit Agent
- Change the `genome` dictionary values to tune strategy
- Modify scoring logic in `_score_sell()`, `_score_take()`, `_score_trade()` methods
- Test changes by relaunching UI

### Backup Your Agent
```powershell
Copy-Item C:\Users\danie\Documents\GitHub\rob311-Project\bazaar-ai\agents\custom_agent.py `
          C:\Users\danie\Documents\GitHub\rob311-Project\agents\custom_agent_backup.py
```

---

## FAQ

**Q: My old batchTest.py doesn't work anymore?**
A: Yes, it uses old `bazaar_ai` imports that conflict with the cloned setup. You can either:
- Update it to use `backend` imports, or
- Use the web UI interface to test instead (recommended for visual feedback)

**Q: Is custom_agent.py my Shark Agent?**
A: Yes, it's your best agent (SharkAgent7 Gen 108) converted to use new `backend` imports. Ready to submit as-is.

**Q: How do I test multiple games programmatically?**
A: Use the web UI for now (visual, immediate feedback). For batch testing, ask to update batchTest.py to backend imports.

**Q: Where are all my files?**
A: 
- Your agents: `agents/` folder (old experiments)
- New submission agent: `bazaar-ai/agents/custom_agent.py`
- Bazaar-AI source: `bazaar-ai/src/`
- Training files: `training_files/` (can still use for reference)
