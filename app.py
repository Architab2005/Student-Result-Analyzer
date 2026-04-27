import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === GLOBAL FUNCTIONS ===
def assign_grade(marks):
    """Updated Grading System as per specification"""
    if marks >= 90:
        return 'O', 'Outstanding'
    elif marks >= 80:
        return 'A+', 'Excellent'
    elif marks >= 75:
        return 'A', 'Distinction'
    elif marks >= 70:
        return 'B+', 'Very Good'
    elif marks >= 60:
        return 'B', 'Good'
    elif marks >= 50:
        return 'C', 'Average'
    elif marks >= 40:
        return 'F', 'Failed'
    elif marks == 0:
        return 'Ab', 'Absent'
    else:
        return 'Ab', 'Absent'

def create_pdf_report(df):
    """Generate professional PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    story.append(Paragraph("📊 Student Performance Analysis Report", styles['Title']))
    story.append(Spacer(1, 20))
    
    # Grading Legend
    story.append(Paragraph("📚 Grading System", styles['Heading2']))
    legend_data = [
        ['Marks', 'Grade', 'Description'],
        ['90-100', 'O', 'Outstanding'],
        ['80-89', 'A+', 'Excellent'],
        ['75-79', 'A', 'Distinction'],
        ['70-74', 'B+', 'Very Good'],
        ['60-69', 'B', 'Good'],
        ['50-59', 'C', 'Average'],
        ['40-49', 'F', 'Failed'],
        ['0', 'Ab', 'Absent']
    ]
    legend_table = Table(legend_data)
    legend_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(legend_table)
    story.append(Spacer(1, 20))
    
    # Metrics Summary
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Students', len(df)],
        ['Average Marks', f"{df['Marks'].mean():.1f}"],
        ['Topper', f"{df.loc[df['Marks'].idxmax(), 'Student']} ({df['Marks'].max():.0f})"],
        ['Pass % (≥50)', f"{len(df[df['Marks']>=50])/len(df)*100:.1f}%"],
        ['At-Risk (<50)', len(df[df['Marks']<50])]
    ]
    table = Table(metrics_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.blue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(table)
    
    # Grade Summary
    story.append(Spacer(1, 20))
    story.append(Paragraph("📊 Grade Distribution", styles['Heading2']))
    summary_data = [['Grade', 'Count']] + \
                   df['Grade'].value_counts().sort_index().reset_index().values.tolist()
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(summary_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# === MAIN APP ===
st.set_page_config(page_title="Student Result Analyzer", layout="wide", page_icon="📊")
st.title("🚀 Ultimate Student Result Analyzer")
st.markdown("**Auto-analysis + Custom Upload (CSV/Excel) + Professional PDF Reports**")

# === SIDEBAR NAVIGATION ===
st.sidebar.markdown("## 📋 Select Module")
selected_module = st.sidebar.selectbox("", 
                                     ["📊 Dashboard", "👥 Students", "🔍 Search", "🤖 Predict", "📄 Custom Upload"])
st.sidebar.markdown("---")
st.sidebar.markdown("**👩‍💻 Archita.B**")
st.sidebar.markdown("*B.Tech CSE '26*")

# === LOAD MAIN DATA ===
@st.cache_data
def load_main_data():
    """Enhanced data loading with ML predictions and insights"""
    df = pd.read_csv("data/Student_Marks.csv")
    df['Student'] = 'S' + df.index.astype(str)
    
    # Dataset columns
    df['Courses'] = df['number_courses']
    df['Study_Hrs'] = df['time_study']
    if 'marks' in df.columns:
        df['Marks'] = df['marks']
    
    df['Marks'] = pd.to_numeric(df['Marks'], errors='coerce')
    df = df.dropna(subset=['Marks']).reset_index(drop=True)
    
    # Grading and ranking
    df[['Grade', 'Grade_Desc']] = df['Marks'].apply(
        lambda x: pd.Series(assign_grade(x))
    )
    df['Rank'] = df['Marks'].rank(ascending=False, method='min').astype(int)
    
    # Study Categories
    df['Study_Category'] = pd.cut(df['Study_Hrs'], 
                                 bins=[0, 2, 4, 6, 10], 
                                 labels=['Low', 'Medium', 'High', 'Very High'])
    
    # ML Prediction Model
    X = df[['Study_Hrs', 'Courses']].fillna(df[['Study_Hrs', 'Courses']].mean())
    y = df['Marks']
    model = LinearRegression()
    model.fit(X, y)
    df['Predicted_Marks'] = model.predict(X)
    df['Prediction_Error'] = abs(df['Marks'] - df['Predicted_Marks'])

    # 🚨 CRITICAL: Store model for Predict module
    df.attrs['model'] = model  
    
    # Smart Insights
    df.attrs['model_r2'] = r2_score(y, df['Predicted_Marks'])
    df.attrs['study_corr'] = df['Study_Hrs'].corr(df['Marks'])
    df.attrs['course_corr'] = df['Courses'].corr(df['Marks'])
    
    return df

# Load main data globally
try:
    df_main = load_main_data()
    os.makedirs('output', exist_ok=True)
    df_main.to_csv('output/analysis.csv', index=False)
    st.sidebar.success("✅ Data loaded!")
except Exception as e:
    df_main = None
    st.sidebar.error(f"🚨 Error: {str(e)}")
    st.sidebar.info("📥 Download Student_Marks.csv from Kaggle")

# === MAIN CONTENT ===
if selected_module == "📊 Dashboard" and df_main is not None:
    st.header("🏆 Main Analysis Dashboard")
    
    # Key Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Average Marks", f"{df_main['Marks'].mean():.1f}")
    col2.metric("🏆 Topper", f"{df_main.loc[df_main['Marks'].idxmax(), 'Student']} ({df_main['Marks'].max():.0f})")
    col3.metric("👥 Total Students", f"{len(df_main):,}")
    col4.metric("🚨 At-Risk (<50)", f"{len(df_main[df_main['Marks']<50]):,}")
    
    # Advanced Insights
    st.markdown("---")
    st.subheader("🔥 Advanced ML Insights")
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        fig_study = px.scatter(df_main, x='Study_Hrs', y='Marks', 
                              size='Courses', color='Grade',
                              title="🔥 Study Hours vs Marks",
                              hover_data=['Student'])
        fig_study.add_hline(y=df_main['Marks'].mean(), line_dash="dash", 
                           line_color="red", annotation_text="Avg Marks")
        st.plotly_chart(fig_study, use_container_width=True)
    
    with col_ins2:
        st.metric("🔗 Study-Marks Corr", f"{df_main.attrs['study_corr']:.3f}")
        st.metric("📚 Course-Marks Corr", f"{df_main.attrs['course_corr']:.3f}")
        st.metric("🤖 ML Accuracy R²", f"{df_main.attrs['model_r2']:.3f}")
        st.info(f"**Best Predictor:** Study Hours")
    
    # Course Load Analysis
    st.subheader("🎯 Course Load Analysis")
    course_analysis = df_main.groupby('Courses').agg({
        'Marks': ['mean', 'count'],
        'Study_Hrs': 'mean'
    }).round(2)
    course_analysis.columns = ['Avg Marks', 'Count', 'Avg Study Hrs']
    st.dataframe(course_analysis, use_container_width=True)
    
    best_course = course_analysis['Avg Marks'].idxmax()
    st.success(f"🏆 **Best:** {best_course} courses (Avg: {course_analysis.loc[best_course, 'Avg Marks']:.1f})")
    
    # Visualizations
    st.subheader("📊 Visual Analysis")
    col_a, col_b = st.columns(2)
    
    with col_a:
        grade_counts = df_main['Grade'].value_counts().sort_index()
        fig1 = px.bar(x=grade_counts.index, y=grade_counts.values,
                     title="📚 Grade Distribution",
                     color=grade_counts.values,
                     color_continuous_scale='RdYlGn_r',
                     height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_b:
        top20 = df_main.nlargest(20, 'Marks')
        fig2 = px.bar(top20, x='Rank', y='Marks', color='Grade',
                     title="🏆 Top 20 Students", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Performance Categories
    st.subheader("🏅 Performance Categories")
    
    st.markdown("### 🌟 **Toppers (O, A+)**")
    toppers = df_main[df_main['Grade'].isin(['O', 'A+'])].sort_values('Marks', ascending=False)
    st.dataframe(toppers[['Rank', 'Student', 'Marks', 'Grade', 'Grade_Desc']], 
                use_container_width=True, hide_index=True)
    
    st.markdown("### ✅ **Above Average (A, B+)**")
    above_avg = df_main[df_main['Grade'].isin(['A', 'B+'])].sort_values('Marks', ascending=False)
    st.dataframe(above_avg[['Rank', 'Student', 'Marks', 'Grade', 'Grade_Desc']], 
                use_container_width=True, hide_index=True)
    
    st.markdown("### ⚠️ **Below Average (C)**")
    below_avg = df_main[df_main['Grade'] == 'C'].sort_values('Marks', ascending=False)
    st.dataframe(below_avg[['Rank', 'Student', 'Marks', 'Grade', 'Grade_Desc']], 
                use_container_width=True, hide_index=True)
    
    st.markdown("### 🚨 **Enhanced At-Risk Students (F, Ab)**")
    at_risk = df_main[df_main['Grade'].isin(['F', 'Ab'])].copy()
    if len(at_risk) > 0:
        at_risk['Risk_Score'] = (60 - at_risk['Marks']) / 60
        risk_cols = ['Student', 'Marks', 'Study_Hrs', 'Courses', 'Predicted_Marks', 'Risk_Score']
        st.dataframe(at_risk[risk_cols].sort_values('Risk_Score', ascending=False), 
                    use_container_width=True)
        st.warning(f"🚨 **Highest Risk:** {at_risk.iloc[0]['Student']} ({at_risk.iloc[0]['Risk_Score']:.0%})")
    else:
        st.success("✅ No at-risk students!")
    
    # ML Predictions
    st.subheader("🤖 ML Prediction Results")
    pred_df = df_main[['Student', 'Marks', 'Predicted_Marks', 'Prediction_Error']].head(15)
    pred_df['Error_%'] = (pred_df['Prediction_Error'] / pred_df['Marks']) * 100
    st.dataframe(pred_df.round(1), use_container_width=True)
    
    # Download
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Download Main PDF Report",
            data=create_pdf_report(df_main),
            file_name=f"student_analysis_{len(df_main)}_students.pdf",
            mime="application/pdf"
        )
    with col_dl2:
        csv_data = df_main.to_csv(index=False).encode()
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"student_results_{len(df_main)}.csv",
            mime="text/csv"
        )

elif selected_module == "📄 Custom Upload":
    st.header("📤 Custom Upload Analyzer")
    st.markdown("**Supports CSV + Excel (.xlsx, .xls)**")
    
    uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls'], key="custom")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_custom = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded CSV: {uploaded_file.name}")
            else:
                df_custom = pd.read_excel(uploaded_file)
                st.success(f"✅ Loaded Excel: {uploaded_file.name}")
            
            student_cols = ['Student', 'student', 'Name', 'name', 'ID']
            marks_cols = ['Marks', 'marks', 'score', 'Score', 'percentage', 'Percentage']
            
            student_col = next((col for col in student_cols if col in df_custom.columns), None)
            marks_col = next((col for col in marks_cols if col in df_custom.columns), None)
            
            if student_col:
                df_custom['Student'] = df_custom[student_col]
            else:
                df_custom['Student'] = 'S' + df_custom.index.astype(str)
            
            if marks_col:
                df_custom['Marks'] = df_custom[marks_col]
            else:
                st.error("❌ No marks column found. Need: Marks, marks, score, or percentage")
                st.stop()
            
            df_custom['Marks'] = pd.to_numeric(df_custom['Marks'], errors='coerce')
            df_custom = df_custom.dropna(subset=['Marks']).reset_index(drop=True)
            df_custom[['Grade', 'Grade_Desc']] = df_custom['Marks'].apply(
                lambda x: pd.Series(assign_grade(x))
            )
            df_custom['Rank'] = df_custom['Marks'].rank(ascending=False, method='min').astype(int)
            
            st.success(f"✅ Processed {len(df_custom)} students")
            
            st.subheader("👀 Data Preview")
            st.dataframe(df_custom[['Student', 'Marks', 'Grade', 'Rank']].head(10))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Average", f"{df_custom['Marks'].mean():.1f}")
            col2.metric("🏆 Topper", f"{df_custom.loc[df_custom['Marks'].idxmax(), 'Student']}")
            col3.metric("🚨 At-Risk", len(df_custom[df_custom['Marks']<50]))
            
            grade_counts = df_custom['Grade'].value_counts()
            fig_pie = px.pie(values=grade_counts.values, names=grade_counts.index,
                            title="📊 Grade Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.subheader("🏅 Ranked Results")
            st.dataframe(df_custom[['Rank', 'Student', 'Marks', 'Grade', 'Grade_Desc']], 
                        use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 PDF Report",
                    data=create_pdf_report(df_custom),
                    file_name=f"custom_analysis_{len(df_custom)}_students.pdf",
                    mime="application/pdf"
                )
            with col_dl2:
                csv_data = df_custom.to_csv(index=False).encode()
                st.download_button(
                    label="📄 Export CSV",
                    data=csv_data,
                    file_name=f"custom_results_{len(df_custom)}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.info("👆 Upload **CSV** or **Excel** file with Student/Marks columns")

elif selected_module == "👥 Students":
    st.header("👥 Student Details")
    if df_main is not None:
        st.dataframe(df_main, use_container_width=True)
    else:
        st.info("📊 Load Dashboard first to see student details")

elif selected_module == "🔍 Search":
    st.header("🔍 🔍 Search Students")
    
    # Search inputs
    search_term = st.text_input("🔍 Search by Student ID, Name, or Grade:", placeholder="S1, O, A+")
    grade_filter = st.selectbox("Filter by Grade:", ["All"] + sorted(df_main['Grade'].unique().tolist()))
    
    if search_term or grade_filter != "All":
        # Filter data
        filtered_df = df_main.copy()
        
        if search_term:
            mask = (
                filtered_df['Student'].str.contains(search_term, case=False) |
                filtered_df['Grade'].str.contains(search_term, case=False) |
                filtered_df['Grade_Desc'].str.contains(search_term, case=False)
            )
            filtered_df = filtered_df[mask]
        
        if grade_filter != "All":
            filtered_df = filtered_df[filtered_df['Grade'] == grade_filter]
        
        if len(filtered_df) > 0:
            st.success(f"✅ Found {len(filtered_df)} matching students")
            
            # Search results table
            search_cols = ['Student', 'Marks', 'Grade', 'Grade_Desc', 'Study_Hrs', 'Courses', 'Rank']
            st.dataframe(filtered_df[search_cols].sort_values('Marks', ascending=False), 
                        use_container_width=True, hide_index=True)
            
            # Quick stats
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("📊 Avg Marks", f"{filtered_df['Marks'].mean():.1f}")
            col_s2.metric("🏆 Top Rank", filtered_df['Rank'].min())
            col_s3.metric("⏱️ Avg Study Hrs", f"{filtered_df['Study_Hrs'].mean():.1f}")
            
            # Export filtered results
            csv_filtered = filtered_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Export Filtered Results",
                data=csv_filtered,
                file_name=f"search_results_{len(filtered_df)}.csv",
                mime="text/csv"
            )
        else:
            st.warning("❌ No students found matching your criteria")
    else:
        st.info("👆 Enter search term or select grade to filter students")

elif selected_module == "🤖 Predict":
    st.header("🤖 🤖 ML Marks Predictor")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Model trained on 200+ students using:**
        - Study Hours (time_study)
        - Number of Courses (number_courses)
        
        **Accuracy:** R² = {:.3f} | Study Corr = {:.3f}
        
        **Predict your marks by entering study hours & courses!**
        """.format(df_main.attrs['model_r2'], df_main.attrs['study_corr']))
    
    # Prediction inputs
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        study_hours = st.slider("📚 Study Hours per week", 0.0, 10.0, 4.0, 0.5)
    with col_p2:
        num_courses = st.slider("📖 Number of Courses", 1, 9, 4)
    
    if st.button("🚀 Predict Marks", type="primary"):
        # Make prediction
        prediction_input = np.array([[study_hours, num_courses]])
        predicted_marks = df_main.attrs['model'].predict(prediction_input)[0]
        predicted_grade, grade_desc = assign_grade(predicted_marks)
        
        # Display results
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("🎯 Predicted Marks", f"{predicted_marks:.1f}")
        col_r2.metric("🏅 Predicted Grade", predicted_grade)
        col_r3.metric("📝 Grade Description", grade_desc)
        
        # Study recommendation
        avg_study = df_main[df_main['Grade'] == predicted_grade]['Study_Hrs'].mean()
        if study_hours < avg_study * 0.8:
            st.warning(f"⚠️ **Recommendation:** Study {avg_study:.1f} hrs/week for better results")
        else:
            st.success("✅ Excellent study habits!")
        
        # Similar students
        similar = df_main[
            (df_main['Study_Hrs'].between(study_hours*0.8, study_hours*1.2)) &
            (df_main['Courses'] == num_courses)
        ].head(5)
        
        if len(similar) > 0:
            st.subheader("👥 Similar Students (Real Data)")
            st.dataframe(similar[['Student', 'Study_Hrs', 'Courses', 'Marks', 'Grade']], 
                        use_container_width=True)
 
    # Model performance chart
    st.subheader("📈 Model Performance")
    fig_perf = px.scatter(df_main, x='Marks', y='Predicted_Marks',
                     title="Actual vs Predicted Marks",
                     labels={'Marks':'Actual Marks', 'Predicted_Marks':'Predicted Marks'},
                     color='Grade', size_max=10)
    fig_perf.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color="red", width=3, dash="dash"),
                  name="Perfect Prediction")
    st.plotly_chart(fig_perf, use_container_width=True)

# === FOOTER ===
st.markdown("---")
st.markdown("""
**👩‍💻 Developed by : Archita.B | B.TECH CSE'26 | Tech‑Driven Business Solutions Enthusiast**
""")
