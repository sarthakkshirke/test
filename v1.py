import streamlit as st
import openai
import pandas as pd
import pdfplumber
from document import Document
import re
import json
import logging
from typing import Any, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, DEPLOYMENT_NAME, VERSION

# Configure OpenAI
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = "azure"
openai.api_version = VERSION

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retry mechanism for OpenAI API calls
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_openai_api(prompt: str) -> dict:
    """Robust OpenAI API call handler with retry logic."""
    try:
        logger.info(f"Sending prompt: {prompt[:200]}...")  # Log truncated prompt
        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            request_timeout=30
        )
        return response
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        st.error("AI service temporarily unavailable. Please try again later.")
        raise

@st.cache_data
def load_data(file_path: str) -> dict:
    """
    Load and validate all sheets from the Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        dict: Contains skills_df, cl_mapping, and pl_cl_mapping.

    Raises:
        ValueError: If any sheet is missing required columns or data is invalid.
    """
    try:
        xls = pd.ExcelFile(file_path)
        
        # Load and validate each sheet
        skills_df = pd.read_excel(xls, "skills")
        cl_df = pd.read_excel(xls, "cl")
        pl_caps_df = pd.read_excel(xls, "pl_caps")
        
        # Validate sheets
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
    """
    Validate and clean the Skills sheet.

    Args:
        skills_df (pd.DataFrame): DataFrame containing skills data.

    Returns:
        pd.DataFrame: Validated and cleaned skills DataFrame.

    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    required_columns = [
        "Role", "Skill Category", "Skill Definition",
        "PL1 (Basic)", "PL2 (Intermediate)", "PL3 (Proficient)",
        "PL4 (Advanced)", "PL5 (Expert)", "Weightage"
    ]
    
    # Clean column names by stripping extra spaces
    skills_df.columns = skills_df.columns.str.strip()
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in skills_df.columns]
    if missing_columns:
        raise ValueError(f"Skills sheet is missing required columns: {missing_columns}")

    # Handle percentage formatting for Weightage
    if skills_df["Weightage"].dtype == object:
        skills_df["Weightage"] = (
            skills_df["Weightage"]
            .str.replace("%", "", regex=False)
            .astype(float)
        )
    elif skills_df["Weightage"].max() <= 1:
        skills_df["Weightage"] = skills_df["Weightage"] * 100

    # Validate weight sums per role
    for role in skills_df["Role"].unique():
        role_skills = skills_df[skills_df["Role"] == role]
        total_weight = role_skills["Weightage"].sum()
        
        if not (95 <= total_weight <= 105):
            skill_list = "\n- ".join([
                f"{row['Skill Category']}: {row['Weightage']}%"
                for _, row in role_skills.iterrows()
            ])
            raise ValueError(
                f"Invalid weights for {role} ({total_weight:.1f}%)\n"
                f"Skill breakdown:\n{skill_list}\n"
                f"Total must be 100% ±5%"
            )

    return skills_df

def validate_cl_sheet(cl_df: pd.DataFrame) -> dict:
    """
    Validate and clean the CL sheet.

    Args:
        cl_df (pd.DataFrame): DataFrame containing CL data.

    Returns:
        dict: Mapping of CL levels to details.

    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    required_columns = ["BAND", "CL", "Managerial", "Functional"]
    
    # Clean column names by stripping extra spaces
    cl_df.columns = cl_df.columns.str.strip()
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in cl_df.columns]
    if missing_columns:
        raise ValueError(f"CL sheet is missing required columns: {missing_columns}")

    # Clean and validate CL values
    cl_df["CL"] = cl_df["CL"].astype(str).str.replace(" ", "").str.upper()
    
    # Create CL to designation mapping
    cl_mapping = {}
    for _, row in cl_df.iterrows():
        cl_mapping[row["CL"]] = {
            "band": row["BAND"],
            "managerial": row["Managerial"],
            "functional": row["Functional"]
        }

    return cl_mapping

def validate_pl_caps_sheet(pl_caps_df: pd.DataFrame) -> dict:
    """
    Validate and clean the PL Caps sheet.

    Args:
        pl_caps_df (pd.DataFrame): DataFrame containing PL Caps data.

    Returns:
        dict: Mapping of PL levels to maximum CL levels.

    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    required_columns = ["PL Level", "Max CL Level"]
    
    # Clean column names by stripping extra spaces
    pl_caps_df.columns = pl_caps_df.columns.str.strip()
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in pl_caps_df.columns]
    if missing_columns:
        raise ValueError(f"PL Caps sheet is missing required columns: {missing_columns}")

    # Clean and validate PL and Max CL values
    pl_caps_df["PL Level"] = pl_caps_df["PL Level"].astype(str).str.upper()
    pl_caps_df["Max CL Level"] = pl_caps_df["Max CL Level"].astype(str).str.replace(" ", "").str.upper()
    
    # Create PL Level to Max CL mapping
    pl_cl_mapping = {}
    for _, row in pl_caps_df.iterrows():
        max_cl = row["Max CL Level"]
        if "&" in max_cl:  # Handle "CL7 & ABOVE" format
            pl_cl_mapping[row["PL Level"]] = int(max_cl.split("CL")[1].split("&")[0].strip())
        else:
            pl_cl_mapping[row["PL Level"]] = int(max_cl[2:])  # Extract numeric part from CL format

    return pl_cl_mapping

def extract_text_from_resume(uploaded_file) -> str:
    """
    Extract text from a resume file (PDF or DOCX).

    Args:
        uploaded_file: Uploaded file object from Streamlit.

    Returns:
        str: Extracted text (truncated to 15,000 characters).

    Raises:
        ValueError: If the file is too large, unsupported, or empty.
    """
    try:
        # Validate file size
        if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
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

        # Validate extracted text
        if len(text) < 100:
            raise ValueError("File appears to be empty or non-textual")

        return text[:15000]  # Truncate for GPT context
    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        raise

def parse_gpt_response(content: str) -> Dict[str, Any]:
    """
    Parse GPT response into a dictionary.

    Args:
        content (str): GPT response content.

    Returns:
        dict: Parsed JSON data.

    Raises:
        ValueError: If no JSON is found or parsing fails.
    """
    try:
        content = content.strip().replace('\\"', '"')
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_str:
            raise ValueError("No JSON found in GPT response")
        return json.loads(json_str.group())
    except Exception as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        raise ValueError("Could not parse AI response")

def analyze_resume_with_gpt(resume_text: str, role: str, skills_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a resume using GPT and assign proficiency levels (PLs) for each skill.

    Args:
        resume_text (str): Extracted text from the resume.
        role (str): Target role for analysis.
        skills_df (pd.DataFrame): DataFrame containing skill definitions and PL descriptions.

    Returns:
        dict: Analysis results including summary, PL levels, and experience years.
    """
    try:
        # Validate role and skills data
        role_skills = skills_df[skills_df["Role"] == role]
        if role_skills.empty:
            raise ValueError(f"No skills defined for role: {role}")

        # Build skill context for GPT
        skill_context = []
        for _, row in role_skills.iterrows():
            skill_context.append(
                f"Skill Category: {row['Skill Category']}\n"
                f"Definition: {row['Skill Definition']}\n"
                f"PL1 (Basic): {row['PL1 (Basic)']}\n"
                f"PL2 (Intermediate): {row['PL2 (Intermediate)']}\n"
                f"PL3 (Proficient): {row['PL3 (Proficient)']}\n"
                f"PL4 (Advanced): {row['PL4 (Advanced)']}\n"
                f"PL5 (Expert): {row['PL5 (Expert)']}\n"
            )

        # Construct GPT prompt
        prompt = f"""Analyze this resume for a {role} position:
{resume_text}

Required Skill Framework:
{''.join(skill_context)}

For each skill category, analyze the candidate's experience and assign a Proficiency Level (PL) based on:
1. Depth of knowledge demonstrated
2. Complexity of tasks described
3. Tools/technologies used
4. Years of experience
5. Alignment with PL descriptions

If a skill category is not mentioned or cannot be correlated with the resume content, assign PL1.

Return JSON with:
{{
  "summary": "Professional summary (100 words)",
  "pl_levels": {{
    "Skill Category": {{
      "level": "PL1-PL5",
      "reason": "Explanation for PL assignment"
    }}
  }},
  "experience_years": "Total relevant experience"
}}"""

        # Call GPT API and parse response
        response = call_openai_api(prompt)
        parsed = parse_gpt_response(response.choices[0].message.content)
        
        # Validate experience
        parsed["experience_years"] = max(0, min(50, int(parsed.get("experience_years", 0))))

        # Ensure all role skills are included in the results
        all_skills = role_skills["Skill Category"].tolist()
        for skill in all_skills:
            if skill not in parsed["pl_levels"]:
                parsed["pl_levels"][skill] = {
                    "level": "PL1",
                    "reason": "No evidence of experience in this area"
                }
        
        return parsed
    except Exception as e:
        logger.error(f"Resume analysis failed: {str(e)}")
        return {"summary": "Analysis error", "pl_levels": {}, "experience_years": 0}

def calculate_career_level(pl_level: str, experience: int, cl_data: Dict[str, Any], pl_caps: Dict[str, int]) -> Dict[str, Any]:
    """
    Calculate the final career level (CL) based on PL and experience.

    Args:
        pl_level (str): Proficiency level (e.g., "PL4").
        experience (int): Total relevant experience in years.
        cl_data (dict): Mapping of CL levels to details.
        pl_caps (dict): Mapping of PL levels to maximum CL levels.

    Returns:
        dict: Career level details including final CL, band, and functional/managerial info.
    """
    try:
        # Get PL-based CL cap
        pl_cl_value = pl_caps.get(pl_level.upper())
        if not pl_cl_value:
            raise ValueError(f"PL level {pl_level} not found in caps")

        # Calculate experience-based CL
        divisor = 2.2 if pl_level.upper() == "PL5" else 2
        exp_cl_value = max(1, min(16, int(experience // divisor)))
        
        # Determine final CL
        final_cl_value = min(pl_cl_value, exp_cl_value)
        final_cl = f"CL{final_cl_value}"
        
        # Get CL details
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
    """
    Calculate the weighted average proficiency level (PL) for a role.

    Args:
        pl_levels (dict): PL levels for each skill category.
        skills_df (pd.DataFrame): DataFrame containing skill definitions and weightages.
        role (str): Target role.

    Returns:
        str: Weighted average PL (e.g., "PL3").
    """
    try:
        role_skills = skills_df[skills_df["Role"] == role]
        pl_values = {"PL1": 1, "PL2": 2, "PL3": 3, "PL4": 4, "PL5": 5}
        
        total_weight = 0
        weighted_sum = 0
        
        for category, pl in pl_levels.items():
            skill_data = role_skills[role_skills["Skill Category"] == category]
            if not skill_data.empty:
                weight = skill_data["Weightage"].values[0]
                total_weight += weight
                weighted_sum += pl_values.get(pl.upper().replace(" ", ""), 1) * weight
                
        if total_weight < 1:
            return "PL1"
            
        average = round(weighted_sum / total_weight)
        return f"PL{min(max(average, 1), 5)}"
    except Exception as e:
        logger.error(f"PL calculation failed: {str(e)}")
        return "PL1"
    
def evaluate_noteworthy_skills(raw_skills: list, role: str, skills_df: pd.DataFrame) -> list:
    """
    Identify unique technical skills not covered by predefined role categories using AI analysis.
    
    Args:
        raw_skills: List of skills extracted from resume
        role: Target job role for evaluation
        skills_df: DataFrame containing role skill definitions
        
    Returns:
        List of noteworthy skills with explanations (max 5)
    """
    try:
        # Validate inputs
        if not raw_skills or not isinstance(raw_skills, list):
            return []
            
        if role not in skills_df["Role"].unique():
            logger.warning(f"Invalid role '{role}' provided for skill evaluation")
            return []

        # Prepare category data with NaN handling
        role_skills = skills_df[skills_df["Role"] == role].fillna("")
        category_names = role_skills["Skill Category"].str.lower().tolist()
        
        # Build keyword list with boundaries to prevent partial matches
        category_keywords = [
            re.escape(word.lower())
            for text in role_skills[["Skill Definition", "PL1 (Basic)"]].values.flatten().tolist()
            for word in str(text).split() if len(word) > 3
        ]
        keyword_pattern = r'\b(' + '|'.join(sorted(set(category_keywords), key=len, reverse=True)) + r')\b'

        # Filter unique skills using precise matching
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

        # Limit input to prevent token overflow
        truncated_skills = unique_skills[:25]  # Keep within GPT context limits
        
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
        
        # Robust JSON extraction
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
    """
    Generate AI-powered hiring recommendation with structured guidance.
    
    Args:
        overall_pl: Final proficiency level (e.g., "PL4")
        role: Target job role
        skills_df: DataFrame containing skill weights
        
    Returns:
        Formatted recommendation string
    """
    try:
        # Validate inputs
        if not overall_pl.startswith("PL") or not overall_pl[2:].isdigit():
            raise ValueError(f"Invalid proficiency level: {overall_pl}")
            
        role_data = skills_df[skills_df["Role"] == role]
        if role_data.empty:
            return "Error: Invalid role specified"

        # Create structured skill summary
        skill_weights = "\n".join([
            f"- {row['Skill Category']}: {row['Weightage']}% priority"
            for _, row in role_data.iterrows()
        ])

        prompt = f"""Create hiring recommendation for {role} candidate (Proficiency Level {overall_pl}).

Skill Priorities:
{skill_weights}

Structure your response:
1. [STRENGTHS] - Top 3 role-specific capabilities
2. [DEVELOPMENT] - Key areas needing improvement
3. [SUITABILITY] - Role fit assessment (1-2 sentences)
4. [ONBOARDING] - Recommended first 90-day focus areas

Use concise bullet points. Avoid markdown. Max 500 characters."""

        response = call_openai_api(prompt)
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return "Error: Unable to generate recommendation. Please try again."
    
def initialize_data() -> dict:
    """
    Load and validate the required data for the evaluation system.

    Returns:
        dict: Contains skills_df, cl_df, pl_caps, and roles.

    Raises:
        Exception: If data loading or validation fails.
    """
    try:
        data = load_data("PL_Ratings123.xlsx")
        skills_df = data["skills"]
        cl_df = data["cl"]
        pl_caps = data["pl_caps"]
        roles = skills_df["Role"].unique().tolist()
        return {
            "skills_df": skills_df,
            "cl_df": cl_df,
            "pl_caps": pl_caps,
            "roles": roles
        }
    except Exception as e:
        logger.error(f"Data initialization failed: {str(e)}")
        st.error(f"System initialization failed: {str(e)}")
        raise

def display_ui(data: dict):
    """
    Render the Streamlit UI components.

    Args:
        data (dict): Contains skills_df, cl_df, pl_caps, and roles.
    """
    # Custom CSS for styling
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

    # Split the layout into two columns
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        # Role selection dropdown
        selected_role = st.selectbox("Select Target Role", data["roles"], key="role_select")

        # Resume uploader
        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"], key="resume_upload")

        # Evaluate button
        if uploaded_file and st.button("Evaluate Candidate", key="evaluate_button"):
            with st.spinner("Analyzing resume..."):
                try:
                    # Process resume
                    resume_text = extract_text_from_resume(uploaded_file)
                    analysis = analyze_resume_with_gpt(resume_text, selected_role, data["skills_df"])
                    
                    # Calculate results
                    overall_pl = compute_weighted_pl(analysis["pl_levels"], data["skills_df"], selected_role)
                    career_info = calculate_career_level(
                        overall_pl,
                        analysis["experience_years"],
                        data["cl_df"],
                        data["pl_caps"]
                    )
                    
                    # Identify noteworthy skills
                    matched_skills = set(analysis["pl_levels"].keys())
                    all_skills = set(data["skills_df"]["Skill Category"].unique())
                    unmatched_skills = list(all_skills - matched_skills)
                    noteworthy = evaluate_noteworthy_skills(unmatched_skills, selected_role, data["skills_df"])
                    
                    # Generate recommendation
                    recommendation = generate_recommendation(overall_pl, selected_role, data["skills_df"])
                    
                    # Store results in session state
                    st.session_state.update({
                        "summary": analysis["summary"],
                        "pl_levels": analysis["pl_levels"],
                        "overall_pl": overall_pl,
                        "career_info": career_info,
                        "noteworthy_skills": noteworthy,
                        "recommendation": recommendation
                    })
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

    # Display results if available
    if "overall_pl" in st.session_state:
        with col1:
            # Overall Proficiency Level
            st.markdown(f"""
            <div class="pl-container">
                <h3>Overall Proficiency Level: {st.session_state.overall_pl}</h3>
                <p>{st.session_state.summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Skill Category Analysis
            st.subheader("Skill Category Analysis")
            
            # Filter skills for the selected role
            role_skills = data["skills_df"][data["skills_df"]["Role"] == selected_role]
            
            # Display PL levels for all role-specific categories
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
            # Career Level Analysis
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

            # Noteworthy Skills
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

            # Recommendation
            st.subheader("Recommendation")
            st.markdown(f"""
            <div class="recommendation-box">
                {st.session_state.recommendation}
            </div>
            """, unsafe_allow_html=True)
            
def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(layout="wide", page_title="PL/CL Evaluation System")
    
    try:
        # Initialize data
        data = initialize_data()
        
        # Display UI
        display_ui(data)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
