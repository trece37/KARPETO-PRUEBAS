import sys
import os
# Add current directory (achilles_trading_bot) to path
sys.path.append(os.getcwd())
print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {sys.path}")

try:
    print("Attempting to import src.brain.api.main...")
    from src.brain.api.main import app
    print("Import SUCCESS")
except Exception as e:
    print(f"Import FAILED: {e}")
    import traceback
    traceback.print_exc()
