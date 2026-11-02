"""Run only the explicitly requested campaign stage (default-free, no implicit fit)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from railvivid.campaign import main

if __name__=='__main__':raise SystemExit(main())
