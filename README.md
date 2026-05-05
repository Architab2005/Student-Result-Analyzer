# 🚀 Student Result Analyzer


**AI-Powered Student Performance Dashboard** with ML predictions, search, and professional PDF reports.

## ✨ Features

### 📊 **Dashboard**
- ML-powered insights (Study vs Marks correlation)
- Course load analysis
- Performance categories (Toppers, At-Risk, etc.)
- Interactive Plotly charts (8+ visualizations)

### 🔍 **Search Module**
- Live search by Student ID, Grade, Description
- Grade-wise filtering
- Export filtered results (CSV)

### 🤖 **ML Predictor**
- Predict marks from Study Hours + Courses
- Model accuracy (R² score)
- Similar students comparison
- Personalized study recommendations

### 📤 **Custom Upload**
- CSV + Excel (.xlsx/.xls) support
- Auto-detect Student/Marks columns
- Instant analysis + PDF/CSV export

### 📥 **Professional Reports**
- PDF reports with grading legend
- Metrics summary + grade distribution
- Ready for presentations/printing


## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run App
```bash
streamlit run app.py
```
## 🛠️ Tech Stack
Frontend: Streamlit + Plotly + CSS
Backend: Pandas + Scikit-learn
Reports: ReportLab (PDF)
Data: CSV/Excel
ML: Linear Regression (R²>=0.95)

## 📁 Project Structure
Student result analyzer/
├── app.py              
├── requirements.txt    
├── datasets/           
│   └── Student_Marks.csv  
├── output/             
│   └── analysis.csv   
└── README.md         

## 🔮 ML Model Details
Input Features:
1. Study Hours (0-10 hrs/week)
2. Number of Courses (1-9 courses)
3. Target: Final Marks (0-100)

Performance:
1. R² Score: ~0.95
2. Study Correlation: ~0.94
3. Perfect for student performance prediction


## 📊 Dataset
 Name : Student Marks Dataset

**Columns:**
- `number_courses`: 1-9 courses
- `time_study`: 0-10 hours/week  
- `marks`: Final score (0-100)

## 📄 License
This project is licensed under the MIT License.


## 👩‍💻 Author
**Archita B**  
*B.Tech CSE '26 *  
---

⭐ **Star this repo if it helped you!** ⭐