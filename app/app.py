"""
SmartHire AI Community Edition
A professional open-source ATS (Applicant Tracking System) platform
"""

import sys
from pathlib import Path
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_text
from src.skill_extractor import extract_skills
from src.scoring import (
    calculate_similarity,
    calculate_skill_match,
    calculate_role_score,
    calculate_ats_score,
    candidate_recommendation,
    skill_gap_severity
)
try:
    from src.category_model import train_category_classifier, predict_resume_category
except ImportError:
    from src.category_model import train_category_classifier

    def predict_resume_category(model, resume_text):
        """Predict a resume category even if the direct helper import is unavailable."""
        if model is None:
            return "Unknown"

        try:
            prediction = model.predict([str(resume_text)])
            return prediction[0]
        except Exception:
            return "Unknown"

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="SmartHire AI - Open Source ATS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
:root {
    --primary-color: #ff4b6e;
    --secondary-color: #2ea043;
    --dark-bg: #0e1117;
    --card-bg: #161b22;
    --border: #30363d;
}

* {
    margin: 0;
    padding: 0;
}

.main {
    background-color: var(--dark-bg);
}

h1, h2, h3, h4, h5, h6 {
    color: #f0f6fc;
    font-weight: 600;
}

p, span, div {
    color: #c9d1d9;
}

[data-testid="metric-container"] {
    background-color: var(--card-bg);
    padding: 25px;
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

[data-testid="stMetricValue"] {
    font-size: 2em;
    font-weight: bold;
    color: var(--primary-color);
}

[data-testid="stMetricLabel"] {
    font-size: 0.9em;
    color: #8b949e;
}

.stDataFrame {
    border-radius: 10px;
    border: 1px solid var(--border);
}

div.stButton > button {
    background-color: var(--primary-color);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #ff3657;
    box-shadow: 0 4px 12px rgba(255, 75, 110, 0.3);
}

div.stDownloadButton > button {
    background-color: var(--secondary-color);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    width: 100%;
}

div.stDownloadButton > button:hover {
    background-color: #259a38;
    box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
}

.stFileUploader {
    border: 2px dashed var(--primary-color);
    border-radius: 10px;
    padding: 20px;
}

.section-divider {
    border-top: 2px solid var(--border);
    margin: 30px 0;
}

.info-box {
    background-color: var(--card-bg);
    border-left: 4px solid var(--primary-color);
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
}

.success-box {
    background-color: rgba(46, 160, 67, 0.1);
    border-left: 4px solid var(--secondary-color);
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
if "classifier_model" not in st.session_state:
    st.session_state.classifier_model = None
    st.session_state.model_accuracy = 0.0

# ==================== UTILITY FUNCTIONS ====================

def find_text_column(df, possible_cols=None):
    """Find the text column in a DataFrame intelligently."""
    if possible_cols is None:
        possible_cols = [
            "Resume_str", "Resume", "resume", "text",
            "Description", "description", "Job Description",
            "job_description", "job_descriptions", "Responsibilities",
            "Content", "content", "Summary", "summary", "Body", "body"
        ]
    
    for col in possible_cols:
        if col in df.columns:
            return col
    
    # Return first object/string column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    
    return df.columns[0] if len(df.columns) > 0 else None


def load_default_datasets():
    """Load default datasets safely."""
    resume_path = ROOT / "data" / "Resume" / "Resume.csv"
    job_desc_path = ROOT / "data" / "job_descriptions.csv"
    monster_path = ROOT / "data" / "monster_com-job_sample.csv"

    resumes = None
    job_desc = None
    monster = pd.DataFrame()

    if resume_path.exists():
        try:
            resumes = pd.read_csv(resume_path, on_bad_lines='skip', encoding='utf-8')
        except Exception:
            resumes = pd.read_csv(resume_path, on_bad_lines='skip', encoding='latin1')
    else:
        sample_resume = ROOT / "data" / "sample_resumes.csv"
        if sample_resume.exists():
            try:
                resumes = pd.read_csv(sample_resume, on_bad_lines='skip', encoding='utf-8')
            except Exception:
                resumes = pd.read_csv(sample_resume, on_bad_lines='skip', encoding='latin1')

    if job_desc_path.exists():
        try:
            job_desc = pd.read_csv(job_desc_path, on_bad_lines='skip', encoding='utf-8')
        except Exception:
            job_desc = pd.read_csv(job_desc_path, on_bad_lines='skip', encoding='latin1')
    else:
        sample_jd = ROOT / "data" / "sample_job_descriptions.csv"
        if sample_jd.exists():
            try:
                job_desc = pd.read_csv(sample_jd, on_bad_lines='skip', encoding='utf-8')
            except Exception:
                job_desc = pd.read_csv(sample_jd, on_bad_lines='skip', encoding='latin1')

    if monster_path.exists():
        try:
            monster = pd.read_csv(monster_path, on_bad_lines='skip', encoding='latin1')
        except Exception:
            monster = pd.DataFrame()

    return resumes, job_desc, monster


def safe_read_csv(uploaded_file):
    """Safely read an uploaded CSV file with encoding handling."""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(uploaded_file, on_bad_lines='skip', encoding=encoding)
        except:
            continue
    
    st.error("Could not read file with any supported encoding. Try a different file.")
    return None


def validate_dataframe(df, required_columns=None):
    """Validate DataFrame and handle issues."""
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < 2:
        return False, f"Dataset too small: {len(df)} rows. Minimum 2 required."
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}"
    
    return True, "OK"


def export_results_csv(df, filename="results.csv"):
    """Generate CSV download link for results."""
    csv = df.to_csv(index=False).encode('utf-8')
    return csv


# ==================== PAGE: DASHBOARD ====================

def page_dashboard():
    """Main dashboard page with system overview."""
    st.title("🧠 SmartHire AI Dashboard")
    st.caption("Professional Open-Source ATS Platform")
    
    # Load datasets
    resumes_df, job_desc_df, monster_df = load_default_datasets()
    
    if resumes_df is None:
        st.error("Failed to load default datasets.")
        return
    
    resume_text_col = find_text_column(resumes_df)
    
    # Initialize or train classifier
    if st.session_state.classifier_model is None and "Category" in resumes_df.columns:
        try:
            st.session_state.classifier_model, st.session_state.model_accuracy = \
                train_category_classifier(resumes_df)
        except Exception as e:
            st.warning(f"Could not train classifier: {e}")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Resumes", f"{len(resumes_df):,}")
    
    with col2:
        if "Category" in resumes_df.columns:
            st.metric("🏷️ Resume Categories", resumes_df["Category"].nunique())
        else:
            st.metric("🏷️ Categories", "N/A")
    
    with col3:
        job_desc_count = len(job_desc_df) if job_desc_df is not None else 0
        monster_count = len(monster_df) if monster_df is not None and not monster_df.empty else 0
        total_jobs = job_desc_count + monster_count
        st.metric("📋 Job Sources", f"{total_jobs:,}")
    
    with col4:
        accuracy_pct = round(st.session_state.model_accuracy * 100, 2)
        st.metric("🤖 Model Accuracy", f"{accuracy_pct}%")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Resume Distribution
    if "Category" in resumes_df.columns:
        st.subheader("Resume Distribution by Category")
        category_counts = resumes_df["Category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        
        fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            title="Resume Count by Category",
            color="Count",
            color_continuous_scale="viridis",
            height=400
        )
        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            hovermode="x unified"
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # WordCloud
    st.subheader("📊 Most Common Keywords in Resumes")
    if resume_text_col and not resumes_df[resume_text_col].empty:
        all_text = " ".join(resumes_df[resume_text_col].astype(str).tolist())
        
        if all_text.strip():
            wc = WordCloud(
                width=1200,
                height=500,
                background_color="black",
                colormap="plasma"
            ).generate(all_text)
            
            fig_wc, ax = plt.subplots(figsize=(15, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)


# ==================== PAGE: RESUME SCREENING ====================

def page_resume_screening():
    """Resume screening with CSV upload support."""
    st.title("📋 Resume Screening")
    
    # Sidebar for data source selection
    st.sidebar.header("📥 Data Sources")
    
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Use Default Datasets", "Upload Custom Datasets"]
    )
    
    # Load resumes
    if data_source == "Use Default Datasets":
        resumes_df, _, _ = load_default_datasets()
        if resumes_df is None:
            st.error("Could not load default datasets.")
            return
    else:
        st.sidebar.subheader("Upload Resume CSV")
        resume_file = st.sidebar.file_uploader("Choose Resume CSV", type=['csv'])
        
        if resume_file:
            resumes_df = safe_read_csv(resume_file)
            if resumes_df is None:
                return
            
            is_valid, msg = validate_dataframe(resumes_df)
            if not is_valid:
                st.sidebar.error(f"Resume data invalid: {msg}")
                return
            
            st.sidebar.success(f"Loaded {len(resumes_df)} resumes")
        else:
            st.sidebar.info("Please upload a resume CSV to continue")
            return
    
    # Find resume text column
    resume_text_col = find_text_column(resumes_df)
    
    if resume_text_col is None:
        st.error("Could not find a text column in the resume dataset")
        return
    
    st.sidebar.write(f"📄 Using column: **{resume_text_col}**")
    
    # Job Description Selection
    st.subheader("🎯 Select Job Description")
    
    jd_source = st.selectbox(
        "Job Description Source",
        ["Manual Input", "Upload CSV", "Default Database"],
        label_visibility="collapsed"
    )
    
    job_description = ""
    
    if jd_source == "Manual Input":
        job_description = st.text_area(
            "Paste or type the job description:",
            height=220,
            placeholder="Enter job requirements, skills, experience...",
            label_visibility="collapsed"
        )
    
    elif jd_source == "Upload CSV":
        jd_file = st.file_uploader("Upload Job Descriptions CSV", type=['csv'], key='jd_upload')
        
        if jd_file:
            jd_df = safe_read_csv(jd_file)
            if jd_df is not None:
                jd_col = find_text_column(jd_df)
                selected_jd_idx = st.selectbox("Select Job Description", jd_df.index, label_visibility="collapsed")
                job_description = str(jd_df.loc[selected_jd_idx, jd_col])
                
                with st.expander("Preview Selected Job Description"):
                    st.text_area("", job_description, height=150, disabled=True, label_visibility="collapsed")
    
    else:  # Default Database
        _, job_desc_df, monster_df = load_default_datasets()
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            job_src = st.radio("Job Description Source", ["Internal DB", "Monster"], label_visibility="collapsed")
        
        with col_b:
            if job_src == "Internal DB":
                if job_desc_df is not None and not job_desc_df.empty:
                    jd_col = find_text_column(job_desc_df)
                    selected_idx = st.selectbox(
                        "Select Internal Job",
                        job_desc_df.index,
                        label_visibility="collapsed"
                    )
                    job_description = str(job_desc_df.loc[selected_idx, jd_col])
            else:
                if monster_df is not None and not monster_df.empty:
                    jd_col = find_text_column(monster_df)
                    selected_idx = st.selectbox(
                        "Select Monster Job",
                        monster_df.index,
                        label_visibility="collapsed"
                    )
                    job_description = str(monster_df.loc[selected_idx, jd_col])
    
    # Filtering Options
    col1, col2 = st.columns(2)
    
    with col1:
        top_n = st.slider("Number of Top Candidates to Show", 5, 100, 10)
    
    with col2:
        category_filter = "All"
        if "Category" in resumes_df.columns:
            category_options = ["All"] + sorted(resumes_df["Category"].dropna().unique().tolist())
            category_filter = st.selectbox("Filter by Category", category_options, label_visibility="collapsed")
    
    # Analyze Button
    if st.button("🔍 Analyze Resumes", width="stretch"):
        if not job_description.strip():
            st.warning("⚠️ Please provide a job description")
            st.stop()
        
        with st.spinner("🔄 Analyzing resumes and ranking candidates..."):
            try:
                # Prepare data
                df = resumes_df.copy()
                
                if category_filter != "All" and "Category" in df.columns:
                    df = df[df["Category"] == category_filter]
                
                if df.empty:
                    st.warning(f"No resumes found in category: {category_filter}")
                    return
                
                # Clean texts
                df["clean_resume"] = df[resume_text_col].astype(str).apply(clean_text)
                clean_jd = clean_text(job_description)
                
                # Extract skills
                required_skills = extract_skills(clean_jd)
                df["resume_skills"] = df["clean_resume"].apply(extract_skills)
                
                # Calculate scores
                similarities = calculate_similarity(df["clean_resume"].tolist(), clean_jd)
                df["similarity_score"] = similarities * 100
                
                df["skill_match_score"] = df["resume_skills"].apply(
                    lambda skills: calculate_skill_match(skills, required_skills)
                )
                
                if "Category" in df.columns:
                    df["role_match_score"] = df["Category"].apply(
                        lambda cat: calculate_role_score(cat, clean_jd)
                    )
                else:
                    df["role_match_score"] = 50.0
                
                df["ats_score"] = df.apply(
                    lambda row: calculate_ats_score(
                        row["similarity_score"],
                        row["skill_match_score"],
                        row["role_match_score"]
                    ),
                    axis=1
                )
                
                # Matched and missing skills
                df["matched_skills"] = df["resume_skills"].apply(
                    lambda skills: list(set(required_skills).intersection(set(skills)))
                )
                df["missing_skills"] = df["resume_skills"].apply(
                    lambda skills: list(set(required_skills) - set(skills))
                )
                
                # Recommendations
                df["recommendation"] = df["ats_score"].apply(candidate_recommendation)
                df["skill_gap_severity"] = df["missing_skills"].apply(skill_gap_severity)
                
                # Prediction of category
                if st.session_state.classifier_model is not None:
                    df["predicted_category"] = df["clean_resume"].apply(
                        lambda x: predict_resume_category(st.session_state.classifier_model, x)
                    )
                else:
                    df["predicted_category"] = "Unknown"
                
                df["reason_for_ranking"] = df.apply(
                    lambda row: (
                        f"Scored {row['ats_score']}% due to "
                        f"{round(row['similarity_score'], 1)}% resume-to-job similarity, "
                        f"{round(row['skill_match_score'], 1)}% skill match, and "
                        f"{round(row['role_match_score'], 1)}% role relevance."
                    ),
                    axis=1
                )
                
                # Rank and save
                ranked_df = df.sort_values(by="ats_score", ascending=False).reset_index(drop=True)
                ranked_df["Rank"] = ranked_df.index + 1
                
                # Save results
                output_path = ROOT / "outputs" / "ranked_candidates.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                ranked_df.to_csv(output_path, index=False)
                
                st.session_state.last_results = ranked_df
                st.session_state.required_skills = required_skills
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
        
        st.success("✅ Analysis Complete!")
        
        # Top Candidate Summary
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🏆 Top Candidate Summary")
        
        top_candidate = ranked_df.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ATS Score", f"{top_candidate['ats_score']:.1f}%")
        with col2:
            st.metric("Similarity", f"{top_candidate['similarity_score']:.1f}%")
        with col3:
            st.metric("Skill Match", f"{top_candidate['skill_match_score']:.1f}%")
        with col4:
            st.metric("Role Match", f"{top_candidate['role_match_score']:.1f}%")
        
        st.info(f"📌 {top_candidate['reason_for_ranking']}")
        
        # Required Skills
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🎯 Required Skills from Job Description")
        
        if required_skills:
            skills_cols = st.columns(4)
            for idx, skill in enumerate(sorted(required_skills)):
                with skills_cols[idx % 4]:
                    st.caption(f"✓ {skill}")
        else:
            st.info("No predefined skills extracted from job description")
        
        # Top Candidates Table
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📊 Top Ranked Candidates")
        
        display_cols = [
            "Rank",
            "predicted_category",
            "ats_score",
            "recommendation",
            "similarity_score",
            "skill_match_score",
            "role_match_score",
            "skill_gap_severity"
        ]
        
        if "Category" in ranked_df.columns:
            display_cols.insert(2, "Category")
        
        st.dataframe(
            ranked_df[display_cols].head(top_n),
            width='stretch',
            height=400
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Comparison")
            fig_scores = px.bar(
                ranked_df.head(top_n),
                x="Rank",
                y="ats_score",
                color="recommendation",
                title="ATS Scores by Rank",
                height=400,
                color_discrete_sequence=px.colors.sequential.RdYlGn
            )
            fig_scores.update_layout(template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_scores, width='stretch')
        
        with col2:
            st.subheader("Skill Gap Analysis")
            gap_counts = ranked_df.head(top_n)["skill_gap_severity"].value_counts().reset_index()
            gap_counts.columns = ["Skill Gap", "Count"]
            
            fig_gap = px.pie(
                gap_counts,
                names="Skill Gap",
                values="Count",
                title="Skill Gap Severity",
                height=400,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_gap.update_layout(template="plotly_dark")
            st.plotly_chart(fig_gap, width='stretch')
        
        # Detailed Explanation
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📋 Detailed Candidate Analysis")
        
        selected_rank = st.selectbox(
            "Select a candidate to view details",
            ranked_df["Rank"].head(top_n)
        )
        
        selected_candidate = ranked_df[ranked_df["Rank"] == selected_rank].iloc[0]
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write(f"**Recommendation:** {selected_candidate['recommendation']}")
            st.write(f"**Category:** {selected_candidate.get('predicted_category', 'N/A')}")
            st.write(f"**Skill Gap:** {selected_candidate['skill_gap_severity']}")
        
        with detail_col2:
            st.write(f"**Matched Skills ({len(selected_candidate['matched_skills'])}):**")
            st.write(", ".join(selected_candidate['matched_skills']) if selected_candidate['matched_skills'] else "None")
        
        st.write(f"**Missing Skills ({len(selected_candidate['missing_skills'])}):**")
        st.write(", ".join(selected_candidate['missing_skills']) if selected_candidate['missing_skills'] else "None")
        
        st.write(f"**Analysis:** {selected_candidate['reason_for_ranking']}")
        
        # Download Results
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("⬇️ Export Results")
        
        csv = export_results_csv(ranked_df.head(top_n))
        st.download_button(
            label="📥 Download Top Candidates (CSV)",
            data=csv,
            file_name="ranked_candidates.csv",
            mime="text/csv",
            width='stretch'
        )


# ==================== PAGE: SINGLE RESUME ANALYZER ====================

def page_single_resume():
    """Analyze a single resume against a job description."""
    st.title("🔍 Single Resume Analyzer")
    st.caption("Analyze one resume quickly against a job description")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Resume")
        resume_file = st.file_uploader("Choose a TXT or CSV file", type=['txt', 'csv'], key='single_resume')
        
        resume_text = ""
        if resume_file:
            if resume_file.type == "text/plain":
                resume_text = resume_file.getvalue().decode('utf-8')
            else:
                df = safe_read_csv(resume_file)
                if df is not None:
                    text_col = find_text_column(df)
                    if text_col:
                        resume_text = str(df[text_col].iloc[0])
            
            st.text_area("Resume Content:", resume_text, height=250, disabled=True, label_visibility="collapsed")
    
    with col2:
        st.subheader("📋 Job Description")
        job_text = st.text_area(
            "Paste job description:",
            height=250,
            placeholder="Enter job requirements and description...",
            label_visibility="collapsed"
        )
    
    if st.button("⚡ Analyze", width="stretch"):
        if not resume_text.strip() or not job_text.strip():
            st.warning("⚠️ Please provide both resume and job description")
            st.stop()
        
        try:
            # Clean texts
            clean_resume = clean_text(resume_text)
            clean_job = clean_text(job_text)
            
            # Extract skills
            required_skills = extract_skills(clean_job)
            candidate_skills = extract_skills(clean_resume)
            
            # Calculate scores
            similarity = calculate_similarity([clean_resume], clean_job)[0] * 100
            skill_match = calculate_skill_match(candidate_skills, required_skills)
            role_score = 50  # Default for single resume
            
            ats_score = calculate_ats_score(similarity, skill_match, role_score)
            recommendation = candidate_recommendation(ats_score)
            
            matched_skills = list(set(required_skills).intersection(set(candidate_skills)))
            missing_skills = list(set(required_skills) - set(candidate_skills))
            skill_gap = skill_gap_severity(missing_skills)
            
            # Predict category if available
            predicted_cat = "Unknown"
            if st.session_state.classifier_model is not None:
                predicted_cat = predict_resume_category(st.session_state.classifier_model, clean_resume)
            
            # Display results
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📊 Analysis Results")
            
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            
            with res_col1:
                st.metric("ATS Score", f"{ats_score:.1f}%")
            with res_col2:
                st.metric("Similarity", f"{similarity:.1f}%")
            with res_col3:
                st.metric("Skill Match", f"{skill_match:.1f}%")
            with res_col4:
                st.metric("Category", predicted_cat)
            
            # Recommendation
            color_map = {
                "Strong Hire": "#2ea043",
                "Good Match": "#58a6ff",
                "Moderate Match": "#d29922",
                "Weak Match": "#f85149",
                "Not Recommended": "#8b949e"
            }
            
            color = color_map.get(recommendation, "#8b949e")
            st.markdown(
                f'<div style="background-color: {color}; padding: 20px; border-radius: 8px; text-align: center;">'
                f'<h3 style="color: white; margin: 0;">Recommendation: {recommendation}</h3>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Skills
            col_skills1, col_skills2 = st.columns(2)
            
            with col_skills1:
                st.write("**✅ Matched Skills:**")
                if matched_skills:
                    for skill in matched_skills:
                        st.caption(f"✓ {skill}")
                else:
                    st.caption("None")
            
            with col_skills2:
                st.write("**❌ Missing Skills:**")
                if missing_skills:
                    for skill in missing_skills:
                        st.caption(f"✗ {skill}")
                else:
                    st.caption("None")
            
            st.write(f"**Skill Gap:** {skill_gap}")
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")


# ==================== PAGE: DATA PREVIEW ====================

def page_data_preview():
    """Preview datasets."""
    st.title("📊 Data Preview")
    
    st.subheader("Default Datasets")
    
    resumes_df, job_desc_df, monster_df = load_default_datasets()
    
    tab1, tab2, tab3 = st.tabs(["📄 Resumes", "📋 Job Descriptions", "💼 Monster Jobs"])
    
    with tab1:
        if resumes_df is not None:
            st.write(f"**Total Resumes:** {len(resumes_df)}")
            st.write(f"**Columns:** {', '.join(resumes_df.columns)}")
            st.dataframe(resumes_df.head(10), use_container_width=True)
        else:
            st.error("Could not load resume dataset")
    
    with tab2:
        if job_desc_df is not None:
            st.write(f"**Total Jobs:** {len(job_desc_df)}")
            st.write(f"**Columns:** {', '.join(job_desc_df.columns)}")
            st.dataframe(job_desc_df.head(10), use_container_width=True)
        else:
            st.error("Could not load job descriptions")
    
    with tab3:
        if monster_df is not None and not monster_df.empty:
            st.write(f"**Total Monster Jobs:** {len(monster_df)}")
            st.write(f"**Columns:** {', '.join(monster_df.columns)}")
            st.dataframe(monster_df.head(10), use_container_width=True)
        else:
            st.info("Monster dataset not available")


# ==================== PAGE: ABOUT & OPEN SOURCE ====================

def page_about():
    """About SmartHire AI and open-source information."""
    st.title("ℹ️ About SmartHire AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🌟 SmartHire AI Community Edition
        
        An open-source, professional Applicant Tracking System (ATS) platform built for 
        modern recruitment teams and HR professionals.
        
        ### 🎯 Features
        - **Resume Screening**: Automated ATS scoring using AI
        - **Skill Extraction**: Intelligent skill detection from resumes and job descriptions
        - **Candidate Ranking**: ML-powered candidate ranking system
        - **Category Classification**: Resume category prediction using LinearSVC
        - **Skill Gap Analysis**: Identify missing skills for each candidate
        - **Custom CSV Support**: Upload your own datasets
        - **Single Resume Analyzer**: Analyze individual resumes quickly
        - **Professional UI**: Dark mode dashboard with visualizations
        - **Export Functionality**: Download results as CSV
        
        ### 📊 ATS Scoring Formula
        ```
        ATS Score = (50% Similarity) + (30% Skill Match) + (20% Role Match)
        ```
        
        - **Similarity**: TF-IDF cosine similarity between resume and job description
        - **Skill Match**: Percentage of required skills found in resume
        - **Role Match**: Category relevance score based on job description
        
        ### 🛠️ Technologies Used
        - **Backend**: Python, Streamlit, Pandas, Scikit-Learn
        - **NLP**: TF-IDF Vectorization, Cosine Similarity
        - **ML**: LinearSVC Classification, Train-Test Split
        - **Visualization**: Plotly, Matplotlib, WordCloud
        - **Data**: CSV, UTF-8 and Latin-1 encoding support
        
        """)
    
    with col2:
        st.markdown("""
        ### 📜 License
        **MIT License**
        
        Free for personal and commercial use.
        
        ### 🤝 Contributing
        Contributions are welcome!
        
        Fork on GitHub and submit pull requests.
        
        ### 🚀 Future Roadmap
        - [ ] Resume PDF parsing
        - [ ] Multi-language support
        - [ ] Advanced NLP models
        - [ ] Real-time collaboration
        - [ ] Analytics dashboard
        - [ ] API endpoints
        - [ ] Cloud deployment
        
        ### 📦 Open Source
        This project is fully open-source and 
        available on GitHub.
        
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.subheader("📋 System Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Version", "1.0.0 Community")
    
    with info_col2:
        st.metric("Status", "Active")
    
    with info_col3:
        st.metric("License", "MIT")


# ==================== MAIN APP ====================

def main():
    """Main app router."""
    
    # Sidebar Navigation
    st.sidebar.title("🧠 SmartHire AI")
    st.sidebar.caption("Community Edition v1.0.0")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📋 Resume Screening",
            "🔍 Single Resume Analyzer",
            "👀 Data Preview",
            "ℹ️ About & Open Source"
        ]
    )
    
    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Load and cache classifier
    if st.session_state.classifier_model is None:
        try:
            resumes_df, _, _ = load_default_datasets()
            if resumes_df is not None and "Category" in resumes_df.columns:
                st.session_state.classifier_model, st.session_state.model_accuracy = \
                    train_category_classifier(resumes_df)
        except:
            pass
    
    # Page Router
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "📋 Resume Screening":
        page_resume_screening()
    elif page == "🔍 Single Resume Analyzer":
        page_single_resume()
    elif page == "👀 Data Preview":
        page_data_preview()
    elif page == "ℹ️ About & Open Source":
        page_about()
    
    # Footer
    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.85em; color: #8b949e;'>
    <p>SmartHire AI Community Edition</p>
    <p>Built with ❤️ by the community</p>
    <p><a href='https://github.com' target='_blank'>GitHub</a> | 
    <a href='#' target='_blank'>Docs</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
