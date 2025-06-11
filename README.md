# 🧠 JobSense - Job Role Generation & Salary Prediction using Transformer

JobSense is an AI-powered job analysis tool that uses a GPT-style Transformer model to **generate new job roles** and **predict corresponding salary ranges**. The model is trained on real-world job data and aims to predict trends in employment, useful for career analysts, job seekers, and HR departments.

---

## 📌 Features

- 🔮 Generate **realistic and unique** job roles
- 💸 Predict **average salary** for generated job roles
- ⚙️ Custom **Transformer model** with PyTorch
- 📊 Dataset pre-processing for job roles and salary ranges
- 📈 Epoch-wise training loss tracking

---

## 📁 Project Structure

```bash
JobSense/
│
├── JobSense.ipynb           # Main notebook (model training, generation)
├── requirements.txt         # List of dependencies (auto-created below)
├── best_model.pth           # Saved transformer model weights (not included)
├── README.md                # You're reading this!
```
## 📂 Dataset

This project uses the [Job Description Dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) by **Ravindra Singh Rana**, hosted on Kaggle.

### 🧾 Dataset Overview

- 📄 **Filename**: `Job_Description_Dataset.csv`
- 🧠 **Features**:
  - `Job Title`
  - `Job Description`
  - `Key Skills`
  - `Role Category`
  - `Functional Area`
  - `Industry`
  - `Role`
  - `Salary`
- 🔍 Used columns in this project:
  - **`Role`** (used to generate new job titles)
  - **`Salary`** (used to predict average salary)
### 📌 Example Entry

| Role                          | Salary      |
|------------------------------|-------------|
| Software Development Manager | ₹15,00,000  |
| Data Scientist                | ₹12,00,000  |
| AI Research Engineer          | ₹20,00,000  |

---
## 🔍 Code Walkthrough

The notebook performs the following steps:

### 🔹 **Data Loading & Preprocessing**
- Loads a dataset with job `Role` and `Salary Range`.
- Cleans salary range values and converts them to a **numerical average**.
- Builds a **character-level vocabulary** for tokenization.

### 🔹 **Model Definition**
- Defines a custom Transformer using PyTorch’s `nn.Transformer`.
- Uses **embedding layers**, **positional encoding**, and a **generator head**.
- Creates two heads: one for **job role generation**, and one for **salary prediction**.

### 🔹 **Training Loop**
- Uses **cross-entropy loss** for the character prediction task.
- Trains the model using the **Adam optimizer**.
- Tracks **loss and training time per epoch**.
- Trains over **100 epochs** for optimal performance.

### 🔹 **Generation**
- Performs **autoregressive sampling** from the model to generate new job titles.
- Predicts the **average salary** for the generated role using a separate regression head.

### 🔹 **Model Saving**
- Saves the trained model in `best_model.pth` (ensure you save this manually — not included in the repo).

---

## ⚙️ Installation

> ✅ Python ≥ 3.8 is recommended

### 📦 1. Clone the repository

```bash
git clone https://github.com/yourusername/JobSense.git
cd JobSense
```
### 📦 2. Install dependencies
```bash
pip install -r requirements.txt
```
   If requirements.txt is not present, install manually:
```bash
pip install torch numpy pandas matplotlib scikit-learn
```
📌 Dependencies
Below is the content of requirements.txt:
```bash
torch
numpy
pandas
matplotlib
scikit-learn
```
### 💡 Future Work
✅ Integrate job description generation using NLP

✅ Upgrade tokenizer to word-level or subword-level for higher accuracy

✅ Deploy as a Flask or Streamlit web app

✅ Integrate with real-time job APIs (e.g., LinkedIn, Naukri, etc.)
