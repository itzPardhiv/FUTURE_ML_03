# 🧠 SmartHire AI — Community Edition

<p align="center">
  <strong>An explainable, local-first Applicant Tracking System for resume screening, skill-gap analysis, ATS scoring, and candidate ranking.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Application-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-core-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage-guide">Usage</a> •
  <a href="#-responsible-use">Responsible Use</a>
</p>

---

## 📊 Overview

**SmartHire AI** is an open-source Applicant Tracking System prototype that helps recruiters and hiring teams perform faster and more consistent first-pass resume screening.

The platform combines classical machine learning, natural-language processing, explainable ATS scoring, skill extraction, and interactive analytics inside a polished Streamlit dashboard.

It is intentionally designed to be:

- **Local-first** — runs on a laptop without paid APIs
- **CPU-friendly** — built using TF-IDF, cosine similarity, and LinearSVC
- **Explainable** — shows why a candidate received a particular score
- **Recruiter-friendly** — provides ranked results, skill gaps, recommendations, and CSV export
- **Portfolio-ready** — demonstrates a complete end-to-end machine-learning workflow

> **One-line pitch:** Turn manual resume screening into an efficient, repeatable, and explainable recruitment workflow.

### Built For

- Recruiters and hiring teams
- HR analysts
- Talent-acquisition teams
- Students demonstrating applied NLP and machine-learning skills
- Developers building lightweight ATS prototypes

### Current Demo Scale

| Metric | Current Demo Value |
|---|---:|
| Resume records | 2,484 |
| Resume categories | 24 |
| Job-source records | 1,637,940 |
| Category-classifier accuracy | 71.03% |

> Values depend on the locally available datasets and may change when different files are loaded.

---

## ✨ Core Features

### 🎯 Explainable Resume Screening

- Rank multiple resumes against a selected job description
- Calculate an overall ATS score for every candidate
- Display similarity, skill-match, and role-match components
- Generate an explanation describing why each candidate received their score
- Categorize results using recruiter-friendly recommendations

### 🧩 Skill Intelligence

- Extract skills from resumes and job descriptions
- Identify matched and missing skills
- Calculate skill-match percentages
- Classify skill-gap severity
- Display required skills clearly for recruiter review

### 🤖 Machine-Learning Classification

- Predict resume categories using **TF-IDF + LinearSVC**
- Train the classifier from categorized resume data
- Display model accuracy inside the dashboard
- Support classification across 24 resume categories

### 📋 Bulk Candidate Ranking

- Use built-in resume datasets
- Upload custom resume CSV files
- Filter candidates by category
- Select how many top candidates to display
- Export ranked candidate results as CSV

### 🔍 Single Resume Analyzer

- Upload one TXT or CSV resume
- Compare it directly against a job description
- View ATS score, similarity, skill match, predicted category, and recommendation
- Review matched and missing skills instantly

### 📈 Interactive Analytics

- Resume-category distribution chart
- Dataset metrics and model-performance overview
- Resume keyword WordCloud
- ATS-score comparison chart
- Skill-gap severity visualization
- Interactive Plotly exploration

### 🛡️ Defensive Data Handling

- Multi-encoding CSV support
- Graceful handling of missing datasets
- Automatic text-column detection
- Friendly Streamlit warnings instead of application crashes
- Local export storage under `outputs/`

---

## 📸 Application Preview

### Dashboard

![SmartHire AI Dashboard](visuals/dashboard.png)

### Resume Screening

![Resume Screening](visuals/resume_screening.png)

### Custom Dataset Ranking

![Custom Dataset Ranking](visuals/custom_dataset_ranking.png)

### Single Resume Analyzer

![Single Resume Analyzer](visuals/single_resume_analyzer.png)

---

## 🧮 ATS Scoring Model

SmartHire AI uses a configurable weighted scoring model:

```text
ATS Score = 50% × Similarity Score
          + 30% × Skill Match Score
          + 20% × Role Match Score
```

| Score Component | Description | Range |
|---|---|---:|
| Similarity Score | TF-IDF cosine similarity between resume and job description | 0–100 |
| Skill Match Score | Percentage of required skills identified in the resume | 0–100 |
| Role Match Score | Category relevance heuristic based on the target job | 0–100 |

The weights are normalized automatically and can be modified in:

```text
src/scoring.py
```

### Candidate Recommendations

The final ATS score is converted into a recruiter-friendly recommendation such as:

- Strong Hire
- Good Match
- Moderate Match
- Weak Match
- Not Recommended

These labels are intended to support human review, not replace it.

---

## 🏗️ Architecture

### End-to-End Workflow

```text
Resume CSV / Uploaded Resume
            ↓
      Safe Data Loading
      ├─ Encoding fallback
      ├─ Missing-file handling
      └─ Automatic text-column detection
            ↓
      Text Preprocessing
      ├─ Clean and normalize text
      ├─ Remove noisy characters
      └─ Prepare text for analysis
            ↓
      Skill Intelligence
      ├─ Extract resume skills
      ├─ Extract required job skills
      ├─ Identify matched skills
      └─ Identify missing skills
            ↓
      Machine-Learning Analysis
      ├─ TF-IDF vectorization
      ├─ Cosine similarity
      ├─ LinearSVC category prediction
      └─ Role relevance calculation
            ↓
      ATS Scoring Engine
      ├─ Similarity score
      ├─ Skill-match score
      ├─ Role-match score
      └─ Weighted final ATS score
            ↓
      Recruiter Dashboard
      ├─ Candidate ranking
      ├─ Explainable recommendations
      ├─ Interactive visualizations
      └─ CSV export
```

### Module Responsibilities

| Module | Purpose |
|---|---|
| `app/app.py` | Streamlit UI, navigation, analysis workflow, charts, and exports |
| `src/data_loader.py` | Safe CSV loading and dataset validation |
| `src/preprocessing.py` | Resume and job-description text cleaning |
| `src/skill_extractor.py` | Skill extraction and skill-gap analysis |
| `src/scoring.py` | Similarity, role relevance, ATS scoring, and recommendations |
| `src/category_model.py` | Resume-category classifier training and prediction |
| `scripts/` | Utility and model-training scripts |
| `tests/` | Automated project tests |
| `outputs/` | Generated ranking exports |

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive recruitment dashboard |
| Data processing | Pandas, NumPy | Data loading, cleaning, and manipulation |
| Machine learning | scikit-learn | TF-IDF, cosine similarity, and LinearSVC |
| NLP | TF-IDF and regex-based extraction | Resume matching and skill intelligence |
| Visualization | Plotly | Interactive charts and analytics |
| Keyword analysis | WordCloud, Matplotlib | Resume keyword visualization |
| Testing | pytest | Automated validation |
| Runtime | Python 3.8+ | Core application environment |

---

## 💼 Business Value

### Faster First-Pass Screening

- Reduce repetitive manual resume review
- Rank large candidate pools consistently
- Focus recruiter attention on the most relevant profiles

### Explainable Candidate Evaluation

- Show why a candidate ranked highly or poorly
- Separate similarity, skill match, and role relevance
- Make screening results easier to review and discuss

### Skill-Gap Visibility

- Identify required skills missing from candidate resumes
- Compare candidates against the same job requirements
- Support more structured interview preparation

### Flexible Recruitment Workflow

- Use default datasets for demonstrations
- Upload custom resume datasets
- Enter manual job descriptions or select stored descriptions
- Export ranked candidate lists for further review

---

## 📁 Project Structure

```text
FUTURE_ML_03/
├── .github/                              # GitHub configuration and workflows
├── .venv/                                # Local virtual environment; ignored by Git
│
├── app/
│   ├── __pycache__/                      # Generated Python cache; ignored by Git
│   └── app.py                            # Main Streamlit application
│
├── data/
│   ├── categorized_resumes/              # Categorized resume dataset
│   ├── Resume/
│   │   └── Resume.csv                    # Default resume dataset
│   ├── job_descriptions.csv              # Internal job-description dataset
│   └── monster_com-job_sample.csv        # External job-description sample
│
├── notebooks/                            # Optional experiments and analysis
├── outputs/                              # Generated candidate rankings and exports
├── scripts/                              # Utility and training scripts
│
├── src/
│   ├── __init__.py
│   ├── category_model.py                 # Resume category classifier
│   ├── data_loader.py                    # Dataset loading utilities
│   ├── preprocessing.py                  # Text cleaning and normalization
│   ├── scoring.py                        # ATS scoring logic
│   └── skill_extractor.py                # Skill extraction and gap analysis
│
├── tests/                                # Automated tests
├── visuals/                              # README screenshots and visual assets
├── .gitignore                            # Repository ignore rules
├── LICENSE                               # MIT License
├── main.py                               # Application launcher
├── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies
```

> The structure above preserves the current project architecture. Local virtual environments, caches, generated outputs, large datasets, and private applicant records should not be committed to GitHub.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or later
- Git
- Windows, macOS, or Linux
- Approximately 1 GB of free space, depending on local datasets

### Quick Start

#### 1. Clone the repository

```bash
git clone https://github.com/itzPardhiv/FUTURE_ML_03.git
cd FUTURE_ML_03
```

#### 2. Create a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the application

```bash
streamlit run app/app.py
```

Alternatively:

```bash
python main.py
```

#### 5. Open the application

```text
http://localhost:8501
```

---

## 📊 Usage Guide

### Dashboard

Use the Dashboard to review:

- total resume count;
- number of resume categories;
- available job-source records;
- category-classifier accuracy;
- resume-category distribution;
- common resume keywords.

### Resume Screening

1. Select **Use Default Datasets** or **Upload Custom Datasets**.
2. Enter, upload, or select a job description.
3. Optionally filter resumes by category.
4. Select the number of top candidates to display.
5. Click **Analyze Resumes**.
6. Review the candidate ranking, score breakdown, skill gaps, and recommendations.
7. Export the ranked candidates as CSV.

### Single Resume Analyzer

1. Upload a TXT or CSV resume.
2. Paste a target job description.
3. Click **Analyze**.
4. Review the ATS score, similarity, skill match, category, recommendation, and missing skills.

### Data Preview

Use the Data Preview page to inspect the currently loaded:

- resume dataset;
- internal job-description dataset;
- Monster job-description dataset.

---

## 📄 Expected Dataset Format

SmartHire AI automatically searches for likely text columns such as:

```text
Resume_str
Resume
resume
text
Description
Job Description
job_description
Responsibilities
Content
Summary
Body
```

### Example Resume CSV

```csv
Name,Category,Resume_str
Candidate A,INFORMATION-TECHNOLOGY,"Python developer with SQL and machine-learning experience"
Candidate B,SALES,"Sales professional with lead-generation and customer-service experience"
```

### Example Job-Description CSV

```csv
Role,job_description
AI/ML Intern,"Python, machine learning, SQL, Streamlit, Git, and data visualization required"
```

---

## 🧪 Testing

Run the test suite:

```bash
pytest -q
```

Compile the project to catch syntax issues:

```bash
python -m compileall app src main.py
```

---

## ⚡ Performance Notes

Performance depends heavily on dataset size and local hardware.

For responsive development and demonstrations:

- avoid loading unnecessary multi-million-row datasets;
- cache dataset-loading functions;
- generate WordClouds only when requested;
- train the classifier only when required;
- test custom uploads with smaller sample files first.

---

## 🐛 Troubleshooting

### Blank Streamlit Page

Stop the server and restart it:

```powershell
Ctrl + C
streamlit run app/app.py
```

A blank page may occur when a large dataset, classifier-training process, or WordCloud generation blocks the initial render.

### PowerShell Blocks Environment Activation

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Missing Dataset Warning

Confirm the required files exist under:

```text
data/Resume/Resume.csv
data/job_descriptions.csv
data/monster_com-job_sample.csv
```

The application can also fall back to sample datasets when configured.

### CSV Encoding Error

SmartHire AI attempts multiple encodings, including:

```text
UTF-8
Latin-1
ISO-8859-1
CP1252
```

### Port Already in Use

```bash
streamlit run app/app.py --server.port 8502
```

---

## 🛡️ Responsible Use

SmartHire AI is a **decision-support prototype**. It must not be used as the sole system for accepting or rejecting candidates.

Resume data may contain personal and sensitive information. Users should:

- obtain permission before processing applicant data;
- avoid committing real resumes to public repositories;
- remove unnecessary personally identifiable information;
- review model outputs for errors and bias;
- keep a qualified human reviewer involved in every hiring decision;
- comply with applicable recruitment, privacy, and employment laws.

---

## ⚠️ Current Limitations

- Regex-based skill extraction may miss synonyms and contextual meaning.
- TF-IDF similarity does not fully understand deep semantic relationships.
- Category predictions depend on the quality and balance of training data.
- Role-match scoring is heuristic.
- PDF and DOCX resume parsing are not currently included.
- ATS scores should not be interpreted as objective hiring decisions.

---

## 🔮 Future Roadmap

### Phase 2 — Enhanced Resume Intelligence

- [ ] PDF and DOCX resume parsing
- [ ] Expanded and configurable skill taxonomy
- [ ] Transformer-based semantic similarity
- [ ] Improved role-match scoring
- [ ] Candidate comparison dashboard

### Phase 3 — Platform and API

- [ ] REST API using FastAPI
- [ ] SQLite or PostgreSQL persistence
- [ ] Recruiter authentication
- [ ] Saved jobs and candidate pipelines
- [ ] Cloud deployment

### Phase 4 — Responsible AI and Evaluation

- [ ] Bias and fairness evaluation
- [ ] Model-performance dashboard
- [ ] Configurable recruiter scorecards
- [ ] Audit logs and decision explanations
- [ ] Human-feedback collection

### Phase 5 — Enterprise Features

- [ ] Multi-user collaboration
- [ ] Role-based access control
- [ ] Integration with recruitment platforms
- [ ] Automated reports
- [ ] Multi-language resume support

---

## 🤝 Contributing

Contributions are welcome.

```bash
git checkout -b feature/your-feature
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
```

Then open a pull request describing:

- the problem being solved;
- the proposed implementation;
- testing performed;
- screenshots for UI changes.

Do not include private applicant data, secrets, virtual environments, generated files, or large datasets in pull requests.

---

## 👨‍💻 Author

**A. J. Pardhiv**

Artificial Intelligence and Data Science student focused on machine learning, NLP, data analytics, and practical AI applications.

- GitHub: [@itzPardhiv](https://github.com/itzPardhiv)
- LinkedIn: [AJ Pardhiv](https://www.linkedin.com/in/aj-pardhiv-406a40333)

---

## 📜 License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Streamlit — interactive data-application framework
- scikit-learn — machine-learning algorithms and utilities
- Plotly — interactive visualization engine
- Pandas — data-processing framework
- WordCloud — resume keyword visualization
- Open-source resume and job-description datasets used for development and demonstration

---

## ⭐ Show Your Support

If SmartHire AI is useful to you:

- ⭐ Star the repository
- 🔗 Share it with other developers
- 📝 Submit improvements
- 💬 Open issues with feedback

<p align="center">
  <strong>Built as an explainable, practical, and recruiter-friendly machine-learning portfolio project.</strong>
</p>
