<#
Usage:
  pwsh -File .\tasks.ps1 install
  pwsh -File .\tasks.ps1 install-cpu
  pwsh -File .\tasks.ps1 ner-jsonl
  pwsh -File .\tasks.ps1 train-ner
  pwsh -File .\tasks.ps1 train-ner-dslim
  pwsh -File .\tasks.ps1 train-intent
  pwsh -File .\tasks.ps1 run-demo
  pwsh -File .\tasks.ps1 versions
  pwsh -File .\tasks.ps1 clean
#>

param(
  [Parameter(Mandatory=$true)]
  [ValidateSet(
    "install","install-cpu",
    "ner-jsonl","train-ner","train-ner-dslim",
    "train-intent","run-demo",
    "versions","clean"
  )]
  [string]$Task
)

$ErrorActionPreference = "Stop"

# --- paths ---
$VenvPy = ".\.venv\Scripts\python.exe"
$Pip    = ".\.venv\Scripts\pip.exe"

function Ensure-Venv {
  if (-not (Test-Path ".\.venv")) {
    python -m venv .venv
  }
}

function Run {
  param([string]$Cmd)
  Write-Host ">> $Cmd" -ForegroundColor Cyan
  iex $Cmd
}

switch ($Task) {

  "install" {
    Ensure-Venv
    Run "$Pip install -U pip"
    Run "$Pip install -e '.[ingest,train,llm]'"
    Write-Host "OK. If you need CPU-only torch on Windows:  pwsh -File .\tasks.ps1 install-cpu" -ForegroundColor Yellow
  }

  "install-cpu" {
    Ensure-Venv
    Run "$Pip install torch --index-url https://download.pytorch.org/whl/cpu"
  }

  "ner-jsonl" {
    # Convert Python seed data → JSONL (aligned tokens/labels)
    Ensure-Venv
    Run "$VenvPy -m tools.train.make_ner_jsonl_from_py tools.train.datasets.ner_data --train-out ./data/ner_train.jsonl --eval-out ./data/ner_eval.jsonl"
  }

  "train-ner" {
    # Train from a base encoder (distilbert). For faster convergence on tiny data, use 'train-ner-dslim'.
    Ensure-Venv
    if (-not (Test-Path "./data/ner_train.jsonl")) { Write-Error "Missing ./data/ner_train.jsonl. Run: pwsh -File .\tasks.ps1 ner-jsonl" }
    Run "$VenvPy -m tools.train.ner_cli --train-file .\data\ner_train.jsonl --eval-file .\data\ner_eval.jsonl --base-model distilbert-base-uncased --output-dir .\models\ner-extractor --epochs 12 --batch-size 8"
  }

  "train-ner-dslim" {
    # Train using a NER-tuned base for better results on small data.
    Ensure-Venv
    if (-not (Test-Path "./data/ner_train.jsonl")) { Write-Error "Missing ./data/ner_train.jsonl. Run: pwsh -File .\tasks.ps1 ner-jsonl" }
    Run "$VenvPy -m tools.train.ner_cli --train-file .\data\ner_train.jsonl --eval-file .\data\ner_eval.jsonl --base-model dslim/bert-base-NER --output-dir .\models\ner-extractor --epochs 12 --batch-size 8"
  }

  "train-intent" {
    Ensure-Venv
    if (-not (Test-Path "./data/intent_data.csv")) { Write-Error "Missing ./data/intent_data.csv" }
    Run "$VenvPy -m tools.train.intent_cli .\data\intent_data.csv --output-dir .\models\intent-classifier --epochs 5 --batch-size 16 --val-ratio 0.3"
  }

  "run-demo" {
    Ensure-Venv
    # You can change test3.py to test2.py if desired.
    Run "$VenvPy .\test3.py"
  }

  "versions" {
    Ensure-Venv
    Run "$VenvPy -c ""import sys,transformers,torch; print('python',sys.version); print('transformers',transformers.__version__); print('torch', getattr(torch,'__version__','n/a'))"""
  }

  "clean" {
    # Remove caches and build artifacts (safe)
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Include "*.pyc","*.pyo",".pytest_cache","*.egg-info","dist","build" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned." -ForegroundColor Green
  }
}
