import streamlit as st
import openai
import pandas as pd
import pdfplumber
from docx import Document
import re
import json
import logging
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, DEPLOYMENT_NAME, VERSION

openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = "azure"
openai.api_version = VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_openai_api(prompt: str) -> dict:
    try:
        logger.info(f"Sending prompt: {prompt[:200]}...")  # Log truncated prompt
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            request_timeout=30
        )
        return response
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        st.error("AI service temporarily unavailable. Please try again later.")
        raise

@st.cache_data
def load_data(file_path: str) -> dict:

    try:
        xls = pd.ExcelFile(file_path)
        
        skills_df = pd.read_excel(xls, "skills")
        cl_df = pd.read_excel(xls, "cl")
        pl_caps_df = pd.read_excel(xls, "pl_caps")
        
        skills_df = validate_skills_sheet(skills_df)
        cl_mapping = validate_cl_sheet(cl_df)
        pl_cl_mapping = validate_pl_caps_sheet(pl_caps_df)
        
        return {
            "skills": skills_df,
            "cl": cl_mapping,
            "pl_caps": pl_cl_mapping
        }
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        st.error(f"Data validation error: {str(e)}")
        raise

def validate_skills_sheet(skills_df: pd.DataFrame) -> pd.DataFrame:

    required_columns = [
        "Role", "Skill Category", "Skill Definition",
        "PL1 (Basic)", "PL2 (Intermediate)", "PL3 (Proficient)",
        "PL4 (Advanced)", "PL5 (Expert)", "Weightage"
    ]
    
    skills_df.columns = skills_df.columns.str.strip()
    
    missing_columns = [col for col in required_columns if col not in skills_df.columns]
    if missing_columns:
        raise ValueError(f"Skills sheet is missing required columns: {missing_columns}")

    if skills_df["Weightage"].dtype == object:
        skills_df["Weightage"] = (
            skills_df["Weightage"]
            .str.replace("%", "", regex=False)
            .astype(float)
        )
    elif skills_df["Weightage"].max() <= 1:
        skills_df["Weightage"] = skills_df["Weightage"] * 100

    for role in skills_df["Role"].unique():
        role_skills = skills_df[skills_df["Role"] == role]
        total_weight = role_skills["Weightage"].sum()
        
        if not (98 <= total_weight <= 102):
            skill_list = "\n- ".join([
                f"{row['Skill Category']}: {row['Weightage']}%"
                for _, row in role_skills.iterrows()
            ])
            raise ValueError(
                f"Invalid weights for {role} ({total_weight:.1f}%)\n"
                f"Skill breakdown:\n{skill_list}\n"
                f"Total must be 100% ±2%"
            )

    return skills_df

def validate_cl_sheet(cl_df: pd.DataFrame) -> dict:

    required_columns = ["BAND", "CL", "Managerial", "Functional"]
    
    cl_df.columns = cl_df.columns.str.strip()
    
    missing_columns = [col for col in required_columns if col not in cl_df.columns]
    if missing_columns:
        raise ValueError(f"CL sheet is missing required columns: {missing_columns}")

    cl_df["CL"] = cl_df["CL"].astype(str).str.replace(" ", "").str.upper()
    
    cl_mapping = {}
    for _, row in cl_df.iterrows():
        cl_mapping[row["CL"]] = {
            "band": row["BAND"],
            "managerial": row["Managerial"],
            "functional": row["Functional"]
        }

    return cl_mapping

def validate_pl_caps_sheet(pl_caps_df: pd.DataFrame) -> dict:
    
    required_columns = ["PL Level", "Max CL Level"]
    
    pl_caps_df.columns = pl_caps_df.columns.str.strip()
    
    missing_columns = [col for col in required_columns if col not in pl_caps_df.columns]
    if missing_columns:
        raise ValueError(f"PL Caps sheet is missing required columns: {missing_columns}")

    pl_caps_df["PL Level"] = pl_caps_df["PL Level"].astype(str).str.upper()
    pl_caps_df["Max CL Level"] = pl_caps_df["Max CL Level"].astype(str).str.replace(" ", "").str.upper()
    
    pl_cl_mapping = {}
    for _, row in pl_caps_df.iterrows():
        max_cl = row["Max CL Level"]
        if "&" in max_cl: 
            pl_cl_mapping[row["PL Level"]] = int(max_cl.split("CL")[1].split("&")[0].strip())
        else:
            pl_cl_mapping[row["PL Level"]] = int(max_cl[2:])  

    return pl_cl_mapping

def extract_text_from_resume(uploaded_file) -> str:
    try:
        if uploaded_file.size > 5 * 1024 * 1024: 
            raise ValueError("File size exceeds 5MB limit")

        text = ""
        if uploaded_file.name.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

        if len(text) < 100:
            raise ValueError("File appears to be empty or non-textual")

        return text[:15000]  # Truncate for GPT context
    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        raise

def extract_skills_with_gpt(resume_text: str) -> dict:
    prompt = f"""
    Extract all technical skills, tools, and relevant experience from this resume:
    {resume_text}

    Return a valid JSON object with the structure:
    {{
      "skills": ["skill1", "skill2"],
      "summary": "brief summary"
    }}

    Rules:
    1. The response must be valid JSON
    2. Escape double quotes with a backslash (\\")
    3. No additional text outside JSON
    """

    try:
        response = call_openai_api(prompt)
        content = response.choices[0].message.content
        
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        return json.loads(content.strip())
    
    except Exception as e:
        logger.error(f"Skill extraction failed: {str(e)}")
        return {"skills": [], "summary": "Error extracting skills"}

def map_skills_to_role(role: str, extracted_skills: list, skills_df: pd.DataFrame) -> dict:

    role_skills = skills_df[skills_df["Role"] == role]["Skill Category"].tolist()
    
    prompt = f"""
    Match candidate skills to {role} categories:
    Candidate skills: {extracted_skills}
    Role categories: {role_skills}

    Return JSON with structure:
    {{
      "matched_skills": {{
        "Category1": ["skill1"],
        "Category2": ["skill2"]
      }}
    }}
    """

    try:
        response = call_openai_api(prompt)
        content = response.choices[0].message.content
        
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        return json.loads(content.strip())
    
    except Exception as e:
        logger.error(f"Skill mapping failed: {str(e)}")
        return {"matched_skills": {}}

logger = logging.getLogger(__name__)

def predict_pl_levels(role: str, matched_skills: dict, skills_df: pd.DataFrame) -> dict:

    role_skills = skills_df[skills_df["Role"] == role]

    pl_context = []
    for _, row in role_skills.iterrows():
        pl_context.append(
            f"Skill Category: {row['Skill Category']}\n"
            f"Definition: {row['Skill Definition']}\n"
            f"PL1 (Basic): {row['PL1 (Basic)']}\n"
            f"PL2 (Intermediate): {row['PL2 (Intermediate)']}\n"
            f"PL3 (Proficient): {row['PL3 (Proficient)']}\n"
            f"PL4 (Advanced): {row['PL4 (Advanced)']}\n"
            f"PL5 (Expert): {row['PL5 (Expert)']}\n"
        )

    prompt = f"""
    You are an expert in evaluating proficiency levels (PL) for technical skills.
    Your task is to assign a PL level (PL1-PL5) to each skill category based on the candidate's resume.

    Role: {role}

    Proficiency Level Definitions:
    {''.join(pl_context)}

    Matched Skills (JSON format):
    {json.dumps(matched_skills, indent=2)}

    For each skill category, analyze the candidate's experience and assign a PL level based on:
    1. Depth of knowledge demonstrated
    2. Complexity of tasks described
    3. Tools/technologies used
    4. Years of experience
    5. Alignment with PL descriptions

    If a skill category is not mentioned or cannot be correlated with the resume content, assign PL1.

    Return a JSON object with the structure:
    {{
      "Skill Category": {{
        "level": "PL1-PL5",
        "reason": "Explanation for PL assignment"
      }}
    }}
    """

    try:
        response = call_openai_api(prompt)
        content = response.choices[0].message.content

        logger.info(f"Raw API response: {content}")

        json_str = content
        
        if '```json' in content:
            json_str = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            json_str = content.split('```')[1].split('```')[0].strip()
        
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        
        pl_levels = json.loads(json_str)
        
        if not isinstance(pl_levels, dict):
            raise ValueError("Invalid response structure from GPT")

        return pl_levels

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        logger.error(f"Raw API response content: {content}")
        return {}
    except Exception as e:
        logger.error(f"PL prediction failed: {str(e)}")
        logger.error(f"Raw API response content: {content}")
        return {}
    
def extract_experience_with_gpt(resume_text: str) -> int:
    prompt = f"""
    Extract total years of experience from:
    {resume_text}

    Return JSON format:
    {{"experience_years": number}}
    """

    try:
        response = call_openai_api(prompt)
        data = json.loads(response.choices[0].message.content)
        return int(data.get("experience_years", 0))
    except Exception as e:
        logger.error(f"Experience extraction failed: {str(e)}")
        return 0

def calculate_career_level(pl_level: str, experience: int, cl_data: Dict[str, Any], pl_caps: Dict[str, int]) -> Dict[str, Any]:

    try:
        pl_cl_value = pl_caps.get(pl_level.upper())
        if not pl_cl_value:
            raise ValueError(f"PL level {pl_level} not found in caps")

        divisor = 2.2 if pl_level.upper() == "PL5" else 2
        exp_cl_value = max(1, min(16, int(experience // divisor)))
        
        final_cl_value = min(pl_cl_value, exp_cl_value)
        final_cl = f"CL{final_cl_value}"
        
        cl_info = cl_data.get(final_cl)
        if not cl_info:
            raise ValueError(f"CL {final_cl} not found in CL sheet")
            
        return {
            "pl_based_cl": f"CL{pl_cl_value}",
            "exp_based_cl": f"CL{exp_cl_value}",
            "final_cl": final_cl,
            "band": cl_info["band"],
            "managerial": cl_info["managerial"],
            "functional": cl_info["functional"],
            "experience": experience
        }
    except Exception as e:
        logger.error(f"Career level calculation error: {str(e)}")
        return {
            "pl_based_cl": "Error",
            "exp_based_cl": "Error",
            "final_cl": "Unknown",
            "band": "N/A",
            "managerial": "Unknown",
            "functional": "Unknown",
            "experience": experience
        }

def compute_weighted_pl(pl_levels: Dict[str, Any], skills_df: pd.DataFrame, role: str) -> str:

    try:
        role_skills = skills_df[skills_df["Role"] == role]
        pl_values = {"PL1": 1, "PL2": 2, "PL3": 3, "PL4": 4, "PL5": 5}
        
        total_weight = 0
        weighted_sum = 0
        
        for category, pl_data in pl_levels.items():
            skill_data = role_skills[role_skills["Skill Category"] == category]
            if not skill_data.empty:
                pl_level = pl_data.get("level", "PL1")  
                weight = skill_data["Weightage"].values[0]
                total_weight += weight
                weighted_sum += pl_values.get(pl_level.upper().replace(" ", ""), 1) * weight
                
        if total_weight < 1:
            return "PL1"
            
        average = round(weighted_sum / total_weight)
        return f"PL{min(max(average, 1), 5)}"
    except Exception as e:
        logger.error(f"PL calculation failed: {str(e)}")
        return "PL1"

def evaluate_noteworthy_skills(raw_skills: list, role: str, skills_df: pd.DataFrame) -> list:
    
    try:
        if not raw_skills or not isinstance(raw_skills, list):
            return []
            
        if role not in skills_df["Role"].unique():
            logger.warning(f"Invalid role '{role}' provided for skill evaluation")
            return []

        role_skills = skills_df[skills_df["Role"] == role].fillna("")
        category_names = role_skills["Skill Category"].str.lower().tolist()
        
        category_keywords = [
            re.escape(word.lower())
            for text in role_skills[["Skill Definition", "PL1 (Basic)"]].values.flatten().tolist()
            for word in str(text).split() if len(word) > 3
        ]
        keyword_pattern = r'\b(' + '|'.join(sorted(set(category_keywords), key=len, reverse=True)) + r')\b'

        unique_skills = []
        for skill in raw_skills:
            if not isinstance(skill, str):
                continue
                
            skill_clean = re.sub(r'\s+', ' ', skill.strip()).lower()
            if (skill_clean not in category_names and 
                not re.search(keyword_pattern, skill_clean)):
                unique_skills.append(skill)

        if not unique_skills:
            return []

        truncated_skills = unique_skills[:25]
        
        prompt = f"""Analyze these candidate skills for {role} role:
        {truncated_skills}

            Identify up to 5 most valuable technical skills that:
            1. Are specific tools/technologies (e.g., TensorFlow, Kubernetes)
            2. Not covered by these standard categories: {category_names}
            3. Indicate specialized expertise
            4. Are in high market demand

            Format response as JSON with this structure:
            {{
                "noteworthy_skills": [
                    {{
                        "skill": "exact skill name from input",
                        "reason": "1-line business impact explanation",
                        "relevance": "High/Medium/Low"
                    }}
                ]
            }}"""

        response = call_openai_api(prompt)
        content = response.choices[0].message.content
        
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_str:
            raise ValueError("No valid JSON found in response")
            
        result = json.loads(json_str.group())
        validated_skills = [
            s for s in result.get("noteworthy_skills", [])
            if isinstance(s, dict) and "skill" in s and "reason" in s
        ][:5]

        return validated_skills

    except Exception as e:
        logger.error(f"Skill evaluation failed for {role}: {str(e)}")
        return []

def generate_recommendation(overall_pl: str, role: str, skills_df: pd.DataFrame) -> str:

    try:
        if not overall_pl.startswith("PL") or not overall_pl[2:].isdigit():
            raise ValueError(f"Invalid proficiency level: {overall_pl}")
            
        role_data = skills_df[skills_df["Role"] == role]
        if role_data.empty:
            return "Error: Invalid role specified"

        skill_weights = "\n".join([
            f"- {row['Skill Category']}: {row['Weightage']}% priority"
            for _, row in role_data.iterrows()
        ])

        prompt = f"""Create hiring recommendation for {role} candidate (Proficiency Level {overall_pl}).

        Skill Priorities:
        {skill_weights}

        Structure your response:

        Generate a single paragraph of recommendation for the hiring manager.
        with justified text alignment.
        """
        response = call_openai_api(prompt)
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return "Error: Unable to generate recommendation. Please try again."

def display_ui(data: dict):

    st.markdown("""
    <style>
        .report-title {
            text-align: center;
            color: #2b5876;
            font-size: 2.5rem;
            margin-bottom: 2rem;
        }
        .pl-container {
            background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .skill-card {
            background: white;
            border-left: 4px solid #4a90e2;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            transition: transform 0.2s;
        }
        .skill-card:hover {
            transform: scale(1.02);
        }
        .cl-details {
            background: #e9f5ff;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .recommendation-box {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #eee;
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #357abd;
        }
    </style>
    <h1 class="report-title">AI-Powered Proficiency Evaluation System</h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        selected_role = st.selectbox("Select Target Role", data["skills"]["Role"].unique(), key="role_select")
        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"], key="resume_upload")

        if uploaded_file and st.button("Evaluate Candidate", key="evaluate_button"):
            with st.spinner("Analyzing resume..."):
                try:
                    resume_text = extract_text_from_resume(uploaded_file)                    
                    extracted_data = extract_skills_with_gpt(resume_text)                    
                    matched_skills = map_skills_to_role(selected_role, extracted_data["skills"], data["skills"])                    
                    pl_levels = predict_pl_levels(selected_role, matched_skills, data["skills"])                    
                    overall_pl = compute_weighted_pl(pl_levels, data["skills"], selected_role)                    
                    experience = extract_experience_with_gpt(resume_text)
                    career_info = calculate_career_level(overall_pl, experience, data["cl"], data["pl_caps"])                    
                    matched_skills_list = [skill for category in matched_skills.get("matched_skills", {}).values() for skill in category]
                    unmatched_skills = list(set(extracted_data["skills"]) - set(matched_skills_list))
                    noteworthy_skills = evaluate_noteworthy_skills(unmatched_skills, selected_role, data["skills"])                    
                    recommendation = generate_recommendation(overall_pl, selected_role, data["skills"])
                    
                    st.session_state.update({
                        "summary": extracted_data["summary"],
                        "pl_levels": pl_levels,
                        "overall_pl": overall_pl,
                        "career_info": career_info,
                        "noteworthy_skills": noteworthy_skills,
                        "recommendation": recommendation
                    })
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

    if "overall_pl" in st.session_state:
        with col1:
            st.markdown(f"""
            <div class="pl-container">
                <h3>Overall Proficiency Level: {st.session_state.overall_pl}</h3>
                <p>{st.session_state.summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Skill Category Analysis")           
            role_skills = data["skills"][data["skills"]["Role"] == selected_role]
            
            for _, row in role_skills.iterrows():
                category = row["Skill Category"]
                weight = row["Weightage"]
                details = st.session_state.pl_levels.get(category, {"level": "PL1", "reason": "No evidence of experience in this area"})
                
                st.markdown(f"""
                <div class="skill-card">
                    <b>{category}</b> (Weight: {weight}%)
                    <div style="color: #6c5ce7; font-size: 1.1rem;">
                        {details.get("level", "PL1")}
                    </div>
                    <div style="color: #666; font-size: 0.9rem;">
                        {details.get("reason", "No explanation provided")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("Career Level Analysis")
            ci = st.session_state.career_info
            st.markdown(f"""
            <div class="cl-details">
                <p><b>Experience:</b> {ci['experience']} years</p>
                <p><b>Band:</b> {ci['band']}</p>
                <p><b>Managerial Role:</b> {ci['managerial']}</p>
                <p><b>Functional Role:</b> {ci['functional']}</p>
                <p><b>Final Career Level:</b> {ci['final_cl']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Noteworthy Skills")
            if st.session_state.noteworthy_skills:
                for skill in st.session_state.noteworthy_skills:
                    st.markdown(f"""
                    <div class="skill-card">
                        <div style="font-weight: 500;">{skill['skill']}</div>
                        <div style="color: #666; font-size: 0.9rem;">{skill['reason']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No notable additional skills identified")

            st.subheader("Recommendation")
            st.markdown(f"""
            <div class="recommendation-box">
                {st.session_state.recommendation}
            </div>
            """, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="PL/CL Evaluation System")
    
    try:
        data = load_data("KB.xlsx")        
        display_ui(data)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()