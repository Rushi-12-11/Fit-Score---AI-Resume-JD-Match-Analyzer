import sys
import os
import PyPDF2
import pandas as pd

# Add project root (one level above /app) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import streamlit as st
from src.ats_engine import evaluate_resume


def analyze_resume_format(resume_text: str) -> dict:

    text_lower = resume_text.lower()
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    line_count = len(lines)

    # 1. Detect sections (very simple keyword-based)
    has_summary = any(word in text_lower
                      for word in ["summary", "profile", "objective"])
    has_experience = any(
        phrase in text_lower
        for phrase in ["experience", "work experience", "employment"])
    has_projects = "project" in text_lower  # matches "project" or "projects"
    has_skills = "skill" in text_lower  # matches "skill" or "skills"
    has_education = any(word in text_lower
                        for word in ["education", "b.tech", "btech", "degree"])

    # 2. Bullet point usage
    bullet_count = sum(1 for line in lines if line.startswith(("-", "•", "*")))

    # 3. Basic length check
    # Not too strict: just a rough idea
    if line_count < 15:
        length_flag = "too_short"
    elif line_count > 120:
        length_flag = "too_long"
    else:
        length_flag = "ok"

    # 4. Compute a simple format score (0–100)
    score = 0
    if has_summary:
        score += 15
    if has_experience:
        score += 25
    if has_projects:
        score += 20
    if has_skills:
        score += 20
    if has_education:
        score += 10

    # Bullet usage bonus
    if bullet_count >= 5:
        score += 10
    elif bullet_count >= 1:
        score += 5

    # Length adjustment
    if length_flag == "ok":
        score += 10
    elif length_flag == "too_short":
        score -= 5
    elif length_flag == "too_long":
        score -= 5

    # Clamp score between 0 and 100
    score = max(0, min(100, score))

    return {
        "format_score": score,
        "line_count": line_count,
        "bullet_count": bullet_count,
        "has_summary": has_summary,
        "has_experience": has_experience,
        "has_projects": has_projects,
        "has_skills": has_skills,
        "has_education": has_education,
        "length_flag": length_flag,
    }


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF file object.
    uploaded_file is what we get from st.file_uploader.
    """
    reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # some pages might return None
            extracted_text += page_text + "\n"

    return extracted_text


st.title("Fit Score AI")
st.subheader("Resume–JD Match Analyzer")

st.write(
    "Paste a resume and a job description to get a match score and insights.")

st.markdown("### Resume Input")

# 1. File uploader for optional PDF resume
uploaded_resume = st.file_uploader("Upload Resume PDF (optional)",
                                   type=["pdf"])

if uploaded_resume is not None:
    # 2. If PDF uploaded, extract its text
    extracted = extract_text_from_pdf(uploaded_resume)

    # 3. Show the extracted text in a textarea (user can still edit)
    st.markdown("**Extracted Resume Text (you can edit below if needed):**")
    resume_text = st.text_area("Resume Text", value=extracted, height=200)
else:
    # 4. Fallback: let user paste text manually
    resume_text = st.text_area("Resume Text", height=200)

st.markdown("### Job Description Input")
jd_text = st.text_area("Job Description Text", height=200)

if st.button("Analyze Match"):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please fill both fields.")
    else:
        with st.spinner("Analyzing..."):
            result = evaluate_resume(resume_text, jd_text)
            format_info = analyze_resume_format(resume_text)

        st.success("Done! You can see the results below.")
        st.markdown("### Match Strength")

        score = result["final_score"]
        if score >= 75:
            bar_color = "green"
            st.balloons()
        elif score >= 50:
            bar_color = "orange"
        else:
            bar_color = "red"

        st.progress(score / 100)
        st.markdown(
            f"<p style='color:{bar_color}; font-size:22px; font-weight:bold;'>Score: {score:.2f}</p>",
            unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Overview", "Skills Breakdown", "Details", "Format"])

        with tab1:
            st.markdown("### Overall Fit Summary")
            st.text(result["summary"])

        with tab2:
            st.markdown("### Matched Skills")
            st.write(", ".join(result["matched_skills"]) or "None")

            st.markdown("### Missing Skills (from JD)")
            st.write(", ".join(result["missing_skills"]) or "None")

        # Details Tab: debugging / raw numbers
        with tab3:
            st.markdown("### Raw Scores & Components")
            st.latex(
                r'''\text{Final Score} = (0.7 \times \text{Base Score}) + (0.3 \times \text{Skill Coverage})'''
            )
            st.write(
                f"Base Score (combined): {result['base_score']:.2f} (0.3*TF-IDF + 0.7*Embedding)"
            )
            st.write(f"Skill coverage: {result['skill_coverage']:.2f}%")
            st.markdown("### Raw Components")
            st.write(
                f"TF-IDF base similarity: {result.get('tfidf_base', 0):.2f}")
            st.write(
                f"Embedding base similarity: {result.get('emb_base', 0):.2f}")

        with tab4:
            st.markdown("### Resume Format & Structure")
            st.markdown(f"**Format Score:** {format_info['format_score']}/100")

            # High-level verdict
            fmt_score = format_info["format_score"]
            if fmt_score >= 80:
                st.success(
                    "Good structure. Resume format looks strong for ATS and recruiters."
                )
            elif fmt_score >= 50:
                st.warning(
                    "Average structure. Some improvements can make this resume stronger."
                )
            else:
                st.error(
                    "Weak structure. Resume is missing important sections or is poorly formatted."
                )

            st.markdown("#### Section Checklist")
            st.write(
                f"Summary/Profile section: {'✅' if format_info['has_summary'] else '❌'}"
            )
            st.write(
                f"Experience section: {'✅' if format_info['has_experience'] else '❌'}"
            )
            st.write(
                f"Projects section: {'✅' if format_info['has_projects'] else '❌'}"
            )
            st.write(
                f"Skills section: {'✅' if format_info['has_skills'] else '❌'}")
            st.write(
                f"Education section: {'✅' if format_info['has_education'] else '❌'}"
            )

            st.markdown("#### Length & Bullets")
            st.write(f"Total non-empty lines: {format_info['line_count']}")
            st.write(f"Bullet point lines: {format_info['bullet_count']}")

            length_flag = format_info["length_flag"]
            if length_flag == "ok":
                st.write("Length: ✅ Looks reasonable.")
            elif length_flag == "too_short":
                st.write(
                    "Length: ⚠ Resume seems quite short. Consider adding more detail."
                )
            else:
                st.write(
                    "Length: ⚠ Resume seems long. Consider tightening content."
                )

st.markdown("---")
st.markdown("### Recruiter Mode: Upload and Rank Multiple Resumes")

uploaded_resumes = st.file_uploader("Upload multiple resumes as PDF",
                                    type=["pdf"],
                                    accept_multiple_files=True)

if st.button("Analyze Uploaded Resumes"):
    if not jd_text.strip():
        st.warning(
            "Please paste a Job Description above before analyzing resumes.")
    elif not uploaded_resumes:
        st.warning("Please upload at least one resume PDF.")
    else:
        bulk_results = []

        with st.spinner("Analyzing uploaded resumes..."):
            for file in uploaded_resumes:
                # 1. Extract text from each PDF
                text = extract_text_from_pdf(file)

                if not text.strip():
                    # if extraction fails / empty, skip or record as 0
                    continue

                # 2. Evaluate this resume against the JD
                res = evaluate_resume(text, jd_text)

                # 3. Store result with file name
                bulk_results.append({
                    "File Name":
                    file.name,
                    "Score":
                    res["final_score"],
                    "Skill Coverage":
                    res["skill_coverage"],
                    "Matched Skills":
                    ", ".join(res["matched_skills"]) or "None"
                })

        if not bulk_results:
            st.error(
                "Could not extract text from any of the uploaded resumes.")
        else:
            # 4. Sort by score, highest first
            bulk_results_sorted = sorted(bulk_results,
                                         key=lambda x: x["Score"],
                                         reverse=True)

            st.success(f"Analyzed {len(bulk_results_sorted)} resumes ✅")
            st.markdown("#### Ranked Candidates (Uploaded Files)")

            st.table(bulk_results_sorted)

            df = pd.DataFrame(bulk_results_sorted)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Results as CSV",
                               data=csv,
                               file_name="fitscore_ranking.csv",
                               mime="text/csv")
