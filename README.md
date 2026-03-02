# Auto Tagging Ticket Support System

An intelligent support ticket classification system that automatically assigns tags/categories to customer support tickets using Machine Learning.

---

## Objectives

1. Build a Python Data Analytics project with UI Dashboard and Backend
2. Deploy a Simple ML Model using Python libraries to GCP (Cloud Run)

---

## Business Flow

```
Customer submits ticket
        ↓
Text Preprocessing (cleaning, tokenization, lemmatization)
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression Classifier
        ↓
Predicted Tag + Confidence Score
        ↓
Dashboard displays result to support agent
```

---

## Scope

- **Input**: Raw support ticket text
- **Output**: Predicted category/tag + confidence score
- **Categories**: Based on dataset labels (e.g., Technical Issue, Billing, Account, Shipping, etc.)
- **Model**: Classical ML — TF-IDF + Logistic Regression
- **UI**: Streamlit dashboard (4 tabs: Home, Demo, EDA, Model Performance)
- **Deployment**: GCP Cloud Run via Docker

---

## Project Structure

```
ticket-support/
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Preprocessed/cleaned dataset
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_model_training.ipynb # Model training & evaluation
├── src/
│   ├── preprocessing.py        # Text cleaning & TF-IDF vectorization
│   ├── train.py                # Training pipeline
│   └── predict.py              # Inference/prediction module
├── models/
│   └── model.pkl               # Saved model artifact (joblib)
├── app/
│   └── main.py                 # Streamlit dashboard
├── Dockerfile
├── requirements.txt
├── cloudbuild.yaml             # GCP Cloud Build configuration
├── .env.example
└── README.md
```

---

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Language        | Python 3.11                         |
| ML Framework    | scikit-learn (TF-IDF + LogReg)      |
| NLP             | NLTK (tokenization, stopwords, lemmatization) |
| UI Framework    | Streamlit                           |
| Visualization   | Plotly, Matplotlib, WordCloud       |
| Containerization| Docker                              |
| Cloud Platform  | GCP Cloud Run                       |
| CI/CD           | GCP Cloud Build                     |

---

## Dataset

**Primary**: [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

- Download the dataset from Kaggle
- Place the CSV file in `data/raw/`
- Expected filename: `customer_support_tickets.csv`

---

## Setup & Installation

### 1. Clone and install dependencies

```bash
cd ticket-support
pip install -r requirements.txt
```

### 2. Download NLTK resources

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### 3. Place dataset in data/raw/

Download `customer_support_tickets.csv` from Kaggle and place it in `data/raw/`.

### 4. Train the model

```bash
python src/train.py
```

This will save the trained model to `models/model.pkl`.

### 5. Run Streamlit dashboard locally

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

---

## Docker (Local)

```bash
# Build image
docker build -t ticket-tagger .

# Run container
docker run -p 8080:8080 ticket-tagger

# Open http://localhost:8080
```

---

## GCP Deployment

### Prerequisites
- GCP Project with billing enabled
- APIs enabled: Cloud Run, Artifact Registry, Cloud Build, Cloud Monitoring
- `gcloud` CLI installed and authenticated

### Deploy to Cloud Run

```bash
gcloud run deploy ticket-tagger \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 1Gi
```

### Or use Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml
```

---

## Model Performance Targets

- **Accuracy**: > 80%
- **F1-score (weighted)**: > 0.75
- **Inference time**: < 500ms per ticket

---

## Verification / Testing

| Environment | Command | URL |
|-------------|---------|-----|
| Local Streamlit | `streamlit run app/main.py` | `http://localhost:8501` |
| Local Docker | `docker run -p 8080:8080 ticket-tagger` | `http://localhost:8080` |
| GCP Cloud Run | After deploy | GCP-provided URL |

Test with at least 5 sample tickets from different categories to verify predictions.

---

## Monitoring

- **GCP Cloud Logging**: Request logs, error logs
- **GCP Cloud Monitoring**: Latency alerts, error rate alerts
- **Streamlit**: Request counter and prediction log in dashboard

---

## License

MIT License — for educational/portfolio purposes.
