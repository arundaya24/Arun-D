# ARUNA AI

Image classification menggunakan Convolutional Neural Network (CNN) yang dikembangkan dan dilatih menggunakan PyTorch dengan dataset CIFAR-10.

Model dibangun dari awal tanpa menggunakan API AI eksternal.

## Architecture

```text
React (Vite)
     |
     v
FastAPI
     |
     +---- PyTorch Model
     |     ml/artifacts/model.pt
     |
     +---- MySQL
           Prediction History
           Statistics
```

## Features

- Image classification
- Image upload and prediction
- Top-3 predictions with confidence scores
- Prediction history
- Dashboard statistics
- Model training and evaluation
- Training loss and accuracy visualization
- Confusion matrix

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch |
| Image Processing | Torchvision |
| Database | MySQL |
| Dataset | CIFAR-10 |

## Project Structure

```text
aruna-ai/
├── ml/
│   ├── configs/
│   ├── src/
│   ├── scripts/
│   ├── tests/
│   ├── artifacts/
│   └── data/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   ├── tests/
│   └── sql/
├── frontend/
│   └── src/
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── ml-concepts.md
├── requirements.txt
└── .env.example
```

## Installation

Requirements:
- Python 3.10–3.12
- Node.js 18+
- MySQL 8 (optional)
- NVIDIA GPU with compatible drivers (optional)

### 1. Clone Repository

```powershell
git clone https://github.com/USERNAME/aruna-ai.git
cd aruna-ai
```

### 2. Python Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. PyTorch CUDA

For NVIDIA GPU support:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check the environment:

```powershell
python -m ml.scripts.check_env
```

### 4. Environment Configuration

```powershell
copy .env.example .env
```

For MySQL:

```sql
CREATE DATABASE aruna_ai CHARACTER SET utf8mb4;
```

Configure the database connection in `.env`:

```env
DATABASE_URL=mysql+pymysql://USER:PASSWORD@localhost:3306/aruna_ai
```

Alternatively, use SQLite:

```env
DATABASE_URL=sqlite:///./aruna.db
```

## Usage

### Model Training

Run the training script:

```powershell
python -m ml.scripts.run_train
```

The CIFAR-10 dataset will be downloaded automatically.

Training artifacts are saved in `ml/artifacts/`:

- `model.pt`
- `classes.json`
- `metrics.json`
- `training_curves.png`
- `confusion_matrix.png`

### Backend

From the project root:

```powershell
uvicorn backend.app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Application:

```text
http://localhost:5173
```

## Testing

Run backend and ML tests:

```powershell
python -m pytest backend/tests ml/tests
```

## Model Development

The project includes a manual implementation of gradient descent and backpropagation:

```powershell
python -m ml.scripts.gradient_demo
```

For model documentation, see:

- `docs/ml-concepts.md`
- `docs/architecture.md`
- `docs/api.md`

Main implementation files:

```text
transforms.py
dataset.py
model.py
train.py
evaluate.py
```

## Google Colab

Model training can also be performed using Google Colab with GPU runtime.

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/aruna-ai

!pip install -q -r requirements.txt
!python -m ml.scripts.run_train
```

Copy the generated model artifacts back to `ml/artifacts/` on the local machine.

## License

Dataset: CIFAR-10

Reference:

Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009.
