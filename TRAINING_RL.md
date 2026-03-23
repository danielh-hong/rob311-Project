# RL Training Guide

## What This Does

Trains a neural network on GPU to improve Shark Agent 7 using reinforcement learning.

**Files:**
- `training_files/train_shark_rl.py` — Training script (uses PyTorch, GPU)
- `bazaar-ai/agents/custom_agent_rl.py` — Final submission agent (standard library only)
- `bazaar-ai/agents/rl_weights.py` — Auto-generated trained weights

## Setup

### Install PyTorch (for training only)
```powershell
pip install torch numpy
```

Check if GPU available:
```powershell
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## Training

### Run Training
```powershell
cd C:\Users\danie\Documents\GitHub\rob311-Project
.\venv\Scripts\Activate.ps1
cd training_files
python train_shark_rl.py
```

**What happens:**
1. Trains for 500 games against SmartAgent
2. Uses GPU automatically if available
3. Prints progress every 50 games
4. Generates `bazaar-ai/agents/rl_weights.py` with trained weights
5. ~ 30-60 minutes depending on GPU

**Expected:**
- Creates file: `bazaar-ai/agents/rl_weights.py`
- Win rate should be 55%+ against SmartAgent

## Testing

### Visual Testing (Web UI)
```powershell
cd C:\Users\danie\Documents\GitHub\rob311-Project
.\venv\Scripts\Activate.ps1
cd bazaar-ai
.\launch-bazaar-ai.bat
```

Select `CustomAgent` (old one) or `CustomAgent_RL` (new one) to test.

### Batch Testing
```powershell
cd C:\Users\danie\Documents\GitHub\rob311-Project
.\venv\Scripts\Activate.ps1
python batchTest.py
```

Then edit `batchTest.py` to use:
```python
from bazaar_ai.agents.custom_agent_rl import CustomAgent as Agent1
```

## Submission

Copy trained agent to submission:
```powershell
copy bazaar-ai\agents\custom_agent_rl.py bazaar-ai\agents\custom_agent.py
```

Or keep both and submit `custom_agent_rl.py` directly to AutoLab.

**Note:** Make sure `rl_weights.py` is in same folder as `custom_agent_rl.py` when you submit. It will be embedded as a simple Python dict.

## Troubleshooting

### "No GPU found"
- Runs on CPU instead (slower but still works)
- Training will take 2-3 hours on CPU instead of 30 min

### "Import errors for bazaar_ai"
- Make sure you're in the venv: `.\venv\Scripts\Activate.ps1`
- Try: `pip install -r requirements.txt`

### "RL weights file not found"
- Training finished but didn't create output file
- Check console output from training for errors
- If training failed, agent defaults to heuristic-only mode

## How It Works

1. **Before:** Pure heuristic (Shark Agent 7 genome values)
2. **Training:** Neural network learns to score actions better than heuristics
3. **After:** Blends 60% heuristic + 40% RL for robustness

Final agent uses:
- RL policy to evaluate actions
- Shark 7 logic as fallback
- Zero external libraries (passes handout restrictions)
