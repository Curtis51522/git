# Bakery AI System

Multi-agent AI operations system for a medium-sized bakery-cafe.
**30 bread products + 15 beverages | 10 employees | 2 ovens**

## Architecture

| Module | Function | Tech | Status |
|--------|----------|------|--------|
| S1 | YOLO product recognition ? 30-class detection + tray color + FIFO | YOLOv11s, OpenCV | ?? Training (ablation study) |
| S2 | 7-day demand forecasting + weather + holiday integration | XGBoost, VisualCrossing | ? Pending |
| S3 | CP-SAT shift scheduling with demand-aware role coverage | OR-Tools CP-SAT | ? Pending |
| S4 | Web POS ? JWT auth, combo scoring, checkout + receipt | FastAPI, HTML/CSS/JS | ? Live |
| S5 | Multi-agent AI Brain ? 6 agents, DistilBERT intent, LLM synthesis | DeepSeek, DistilBERT | ? Live |

## S1 ? YOLO Recognition (Current Focus)

### Dataset

| Metric | Value |
|--------|-------|
| Classes | 30 (bread + pastry) |
| Train / Val / Test | 32,974 / 1,997 / 1,998 images |
| Total instances | 70,769 |
| Min instances per class | 1,656 (mantequilla) |
| Max instances per class | 7,772 (croissant) |

**Data pipeline:**
1. Merged 7 Roboflow YOLO datasets into unified 30-class format
2. Removed tiramisu + waffle (not in POS); added brownie + chocolate_cake
3. GroundingDINO auto-labeling for brownie (1,440 instances) + chocolate_cake (661 instances)
4. Removed 2,704 low-quality images (<200px dim or <10KB)
5. Augmented 24 under-represented classes (flip, brightness, contrast, blur) to reach >1,500 instances each

### Ablation Study (Running)

| # | Experiment | Optimizer | Epochs | Batch | Augmentation |
|---|-----------|-----------|--------|-------|-------------|
| 1 | Baseline | AdamW | 100 | 16 | Basic (hsv, flip) |
| 2 | + SGD | SGD | 100 | 16 | Basic |
| 3 | + Epochs | SGD | 200 | 8 | Basic |
| 4 | **Proposed** | SGD | 200 | 8 | Full (mixup, multi_scale, label_smoothing) |

All experiments: lr0=0.005, nbs=batch (effective LR identical), imgsz=640, COCO-pretrained YOLOv11s, AMP enabled.

**Training hardware:** NVIDIA RTX 5070 Ti Laptop (12GB VRAM)

### Outputs per experiment
- 
uns/detect/<tag>/weights/best.pt ? best model
- 
uns/detect/<tag>/results.csv ? per-epoch metrics
- 
uns/detect/<tag>/*.png ? confusion matrix, PR/F1 curves
- s1_recognition/<tag>_metrics.json ? per-class AP
- s1_recognition/ablation_results.json ? comparison table

## Quick Start

### Prerequisites
- Python 3.11+
- MySQL 8.0+
- API keys: DeepSeek, VisualCrossing

### Setup
`ash
git clone https://github.com/Curtis51522/git.git
cd git
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API keys and MySQL credentials

# Train DistilBERT intent classifier (one-time, ~3 min)
cd s5_agent
python -c "from intent_classifier import train_intent_classifier; train_intent_classifier()"
cd ..

# Start main server (port 8002)
python main.py
`

Open http://localhost:8002 for the web UI.

### Default Accounts
| Role | Username | Password |
|------|----------|----------|
| Manager | manager | hash123 |
| Staff | staff1 | hash123 |

## S5 Multi-Agent Brain

### Pipeline
`
User Query -> DistilBERT Intent Classifier (8 intents) -> 6-Agent Orchestration -> Arbitrator -> LLM Synthesis
`

### 6 Agents (programmatic, no LLM)
| Agent | Confidence | Data Source |
|-------|-----------|-------------|
| demand | 90% | S2 XGBoost forecast + per-product trend |
| inventory | 95% | MySQL batch_inventory + freshness tracking |
| production | 85% | Baker count x oven capacity |
| staffing | 95% | S3 CP-SAT schedule |
| promo | 85% | 5-dim scoring |
| profit | 50-90% | MySQL transaction history |

### 8 Intents (DistilBERT, 7,764 samples)
| Intent | Agents | Example |
|--------|--------|---------|
| stock_query | all 6 | "How many croissants tomorrow?" |
| comparison_analysis | demand, inventory, profit | "Compare croissant and donut" |
| waste_analysis | demand, inventory | "Why is waste high?" |
| promo_eval | demand, inventory, promo, profit | "How to promo all products?" |
| schedule_audit | staffing | "Check schedule" |
| profit_analysis | demand, inventory, profit | "Profit margin?" |
| cross_source_audit | all 6 | "Full store health check" |
| out_of_scope | none | Jokes, weather |

## Employees
| ID | Name | Role | Secondary |
|----|------|------|-----------|
| E001 | Ali | baker | ? |
| E002 | Mei | cashier | ? |
| E003 | Raj | barista | cashier |
| E004 | Siti | baker | ? |
| E005 | Ahmad | baker | ? |
| E006 | Priya | cashier | ? |
| E007 | Kumar | baker | ? |
| E008 | David | baker | ? |
| E009 | Chen | barista | ? |
| E010 | Fatima | manager | ? |

## Key Design Decisions
- **30 breads via YOLOv11s** detection; 15 beverages are menu-only (no visual recognition)
- **S5 uses 2-tier LLM**: DeepSeek V4 Pro (plan selection) + DeepSeek V4 Flash (synthesis)
- **DistilBERT local** for intent classification (no API cost)
- **All code in English**
