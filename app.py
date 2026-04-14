import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fpdf import FPDF
import io
import datetime
import tempfile
import os
import plotly.graph_objects as go
import streamlit as st



# --- 1. PAGE SETUP & CUSTOM CSS (Montserrat) ---
st.set_page_config(page_title="ALZ DETECT", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bree+Serif&family=Montserrat:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif !important;
        }
        

        /* Sidebar styling - Translucent with gray border */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.7) !important;
            border-right: 2px solid #808080 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #1a1a1a !important;
        }
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stCheckbox label {
            color: #1a1a1a !important;
            font-weight: 500;
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #1a1a1a !important;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }
        
        /* Sidebar radio/checkbox - light background boxes */
        section[data-testid="stSidebar"] .stRadio > div,
        section[data-testid="stSidebar"] .stCheckbox > div {
            background: rgba(255, 255, 255, 0.7) !important;
            border: 1px solid rgba(52, 152, 219, 0.4);
            border-radius: 8px;
            padding: 8px;
            margin: 4px 0;
        }

        /* Sidebar slider */
        section[data-testid="stSidebar"] .stSlider {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            padding: 10px;
        }

        /* Sidebar selectbox */
        section[data-testid="stSidebar"] .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.8) !important;
            border: 1px solid #3498db;
            border-radius: 6px;
        }

        /* Sidebar divider */
        section[data-testid="stSidebar"] hr {
            border-top: 1px solid #3498db;
            opacity: 0.5;
        }

        /* Sidebar checkboxes with custom styling */
        section[data-testid="stSidebar"] .stCheckbox {
            padding: 5px;
            margin: 3px 0;
        }

        /* Hybrid Assessment box - Default styling */
        .hybrid-box {
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .hybrid-low {
            background-color: rgba(39, 174, 96, 0.2);
            border-left: 4px solid #27AE60;
        }
        .hybrid-high {
            background-color: rgba(192, 57, 43, 0.2);
            border-left: 4px solid #C0392B;
        }
        .hybrid-neutral {
            background-color: rgba(52, 152, 219, 0.2);
            border-left: 4px solid #2980B9;
        }
        .hybrid-box h4 {
            margin: 0 0 8px 0;
            font-size: 1rem;
            font-weight: 600;
        }
        .hybrid-box p {
            margin: 0;
            font-size: 0.9rem;
        }

        /* Score badge */
        .score-badge {
            display: inline-block;
            background: #2C3E50;
            color: white !important;
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-left: 8px;
        }

        /* Main Background with image */
        .stApp {
            background-image: url('https://ik.imagekit.io/kwenoxjg0/Gemini_Generated_Image_hs3q4whs3q4whs3q.png');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
        }

        /* Background overlay for readability */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.2);
            pointer-events: none;
            z-index: -1;
        }

        /* Enhanced main content area - Translucent for readability */
        .block-container {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(52, 152, 219, 0.3);
        }

        /* Enhanced headers with DARK text */
        h1, h2, h3, h4 {
            color: #1a1a1a !important;
            font-weight: 700;
            text-shadow: none;
        }

        /* Main page text - DARK colors */
        .stMarkdown {
            color: #1a1a1a;
            font-family : "Bree Serif"
        }
        
        .stMarkdown p {
            color: #2C3E50 !important;
            font-weight: 500;
        }

        /* Make all text visible on light container */
        div[data-testid="stMarkdownContainer"] p {
            color: #1a1a1a !important;
        }

        /* Label visibility - DARK */
        .stRadio label, .stCheckbox label {
            color: #1a1a1a !important;
            font-weight: 500;
        }

        /* Text input and selects - dark styling */
        .stTextInput > div > div, .stSelectbox > div > div {
            background: rgba(30, 45, 60, 0.95) !important;
            border: 1px solid #3498db;
            color: #ffffff !important;
        }

        /* File uploader - dark styling */
        .stFileUploader {
            background: rgba(30, 45, 60, 0.85);
            border-radius: 12px;
            padding: 20px;
            border: 2px dashed #3498db;
        }

        /* Caption text - LIGHT */
        .stCaption {
            color: #b0b0b0 !important;
        }

        /* Styled containers - LIGHT */
        div[data-testid="stMetric"], 
        div[data-testid="stSuccess"],
        div[data-testid="stInfo"] {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(52, 152, 219, 0.3);
        }

        /* Metric text - DARK - GLOBAL DEFAULT */
        div[data-testid="stMetric"] label {
            color: #1a1a1a !important;
        }


        [data-testid="stMetricValue"], .stMetric [data-testid="metric-value"] {
            color: #1a1a1a !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.3 !important;
            max-width: none !important;
        }


        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #27AE60 !important;
        }
        /* Ensure metric containers don't constrain width */
        div[data-testid="stMetric"] {
            min-width: 0 !important;
        }

        /* Button styling */
        div.stButton > button {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
        }

        /* Divider styling */
        hr {
            border-top: 2px solid rgba(52, 152, 219, 0.4);
            margin: 1.5rem 0;
        }

        /* Radio and checkbox styling - DARK */
        .stRadio > div, .stCheckbox > div {
            background: rgba(30, 50, 70, 0.8);
            padding: 12px;
            border-radius: 10px;
            border: 1px solid rgba(52, 152, 219, 0.3);
        }

        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #3498db, #2ecc71);
        }

        /* Success/Error/Info message styling */
        .stSuccess > div {
            background: rgba(39, 174, 96, 0.25);
            color: #ffffff;
            border-left: 4px solid #2ecc71;
            border-radius: 8px;
        }
        
        .stError > div {
            background: rgba(192, 57, 43, 0.25);
            color: #ffffff;
            border-left: 4px solid #e74c3c;
            border-radius: 8px;
        }

        /* Spinner */
        .stSpinner {
            color: #3498db;
        }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
        }
        ::-webkit-scrollbar-thumb {
            background: #3498db;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


# --- 2. SESSION STATE INITIALIZATION ---
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "result" not in st.session_state:
    st.session_state.result = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "mri_image" not in st.session_state:
    st.session_state.mri_image = None
if "saved_orientation" not in st.session_state:
    st.session_state.saved_orientation = None
if "saved_memory_recall" not in st.session_state:
    st.session_state.saved_memory_recall = None
if "saved_executive_fn" not in st.session_state:
    st.session_state.saved_executive_fn = None


# --- 3. LOAD MODEL ---
@st.cache_resource
def load_alz_model():
    return load_model('my_model.h5')

model = load_alz_model()


# --- 4. MRI GATEKEEPER ---
def validate_mri(img_array):
    img = img_array[0]
    channel_variance = np.var(img, axis=2)
    is_color = np.mean(channel_variance) > 0.01

    top_left_corner = img[0:15, 0:15]
    corner_brightness = np.mean(top_left_corner)
    is_dark_bg = corner_brightness < 0.15

    if is_color:
        return False, "Image rejected: Uploaded image appears to be in full color. Please provide a grayscale MRI scan."
    if not is_dark_bg:
        return False, "Image rejected: Image lacks the dark background typical of a standard MRI."
    return True, "Valid MRI detected."


# --- 5. COGNITIVE HEALTH SCORE CALCULATOR ---
def calculate_cognitive_score(orientation, memory_recall, executive_fn):
    score = 0
    score += 3 if orientation == "Yes" else 0
    score += 4 if memory_recall == "Pass" else 0
    score += 3 if executive_fn == "Pass" else 0
    return score


# --- 6. FUSION ENGINE ---
def fusion_engine(ai_result, cog_score, age, family_history, cog_test_done):
    if not cog_test_done:
        return (
            "Hybrid Assessment Unavailable",
            "No cognitive test data was provided. Complete the Mini-Cognitive Test in the sidebar to unlock the Hybrid Assessment.",
            "hybrid-neutral"
        )

    mild_cases = ["Very Mild Demented", "Mild Demented"]

    if ai_result in mild_cases and cog_score == 10 and age < 65:
        return (
            "🟢 Low Risk",
            "MRI anomalies may be age-related; cognitive function is fully intact.",
            "hybrid-low"
        )
    if ai_result == "Very Mild Demented" and cog_score < 5 and family_history == "Yes":
        return (
            "🔴 High Risk",
            "Severe cognitive decline detected alongside MRI atrophy. Immediate clinical review required.",
            "hybrid-high"
        )
    if ai_result == "Non Demented" and cog_score >= 8:
        return (
            "🟢 Low Risk",
            "No neurological atrophy detected and cognitive function scores are strong. Continue routine preventative care.",
            "hybrid-low"
        )
    if ai_result == "Moderate Demented" or cog_score <= 3:
        return (
            "🔴 High Risk",
            "Significant MRI atrophy and/or severely impaired cognitive performance. Urgent neurological consultation is strongly recommended.",
            "hybrid-high"
        )
    return (
        "⚠️ Moderate Risk",
        f"AI prediction is '{ai_result}' with a Cognitive Health Score of {cog_score}/10. Consider scheduling a follow-up clinical evaluation within 3-6 months.",
        "hybrid-neutral"
    )


# --- 7. COGNITIVE DECLINE RISK LABEL ---
def cognitive_risk_label(cog_score):
    if cog_score >= 8:
        return "Low", int((cog_score / 10) * 100)
    elif cog_score >= 5:
        return "Moderate", int((cog_score / 10) * 100)
    else:
        return "High", int((cog_score / 10) * 100)


# --- 8. RADAR CHART SCORE CALCULATOR ---
def calculate_radar_scores(result, confidence, age, family_history, cog_score, risk_factors):
    genetics     = 90 if family_history == "Yes" else 10
    vascular_count = sum(1 for v in risk_factors.values() if v)
    vascular     = min(vascular_count * 33, 100)
    cognition    = round((1 - cog_score / 10) * 100) if cog_score is not None else 50
    atrophy_map  = {
        "Non Demented":       10,
        "Very Mild Demented": 40,
        "Mild Demented":      70,
        "Moderate Demented":  95,
    }
    brain_atrophy = atrophy_map.get(result, 50)
    age_risk      = round(min(age, 100))

    return {
        "Genetics":      genetics,
        "Vascular":      vascular,
        "Cognition":     cognition,
        "Brain Atrophy": brain_atrophy,
        "Age":           age_risk,
    }


# --- 9. STATE-WISE HOSPITAL DATABASE ---
INDIA_HOSPITALS = {
    "Telangana": [
        {
            "name": "NIMS - Nizam's Institute of Medical Sciences",
            "location": "Punjagutta, Hyderabad, Telangana",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 8:00 AM - 4:00 PM",
            "phone": "+91 40 2348 9000",
            "note": "Premier government neuroscience institute with a dedicated Neurology & Memory Disorders department."
        },
        {
            "name": "Apollo Hospitals - Neurology Centre",
            "location": "Jubilee Hills, Hyderabad, Telangana",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "+91 40 2360 7777",
            "note": "Multi-speciality hospital with an advanced Brain & Spine Institute and dementia care programmes."
        },
        {
            "name": "Yashoda Hospitals - Neurosciences",
            "location": "Secunderabad, Hyderabad, Telangana",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 40 4567 4567",
            "note": "Specialised neurology wing offering memory assessments, cognitive rehabilitation, and dementia management."
        },
        {
            "name": "KIMS Hospital - Department of Neurology",
            "location": "Secunderabad, Hyderabad, Telangana",
            "rating": "4.5 / 5.0",
            "timings": "Mon - Sat: 8:30 AM - 5:30 PM",
            "phone": "+91 40 4488 5000",
            "note": "Krishna Institute of Medical Sciences offers comprehensive neurological evaluations and Alzheimer's care."
        },
    ],
    "Karnataka": [
        {
            "name": "NIMHANS - National Institute of Mental Health and Neurosciences",
            "location": "Hosur Road, Bengaluru, Karnataka",
            "rating": "4.8 / 5.0",
            "timings": "Mon - Sat: 8:00 AM - 4:00 PM",
            "phone": "+91 80 2699 5000",
            "note": "India's top neuroscience institute with a dedicated Cognitive Disorders Clinic and memory research unit."
        },
        {
            "name": "Manipal Hospital - Neurology",
            "location": "Old Airport Road, Bengaluru, Karnataka",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "+91 80 2502 4444",
            "note": "Comprehensive brain health services including neuroimaging, cognitive testing, and dementia care."
        },
        {
            "name": "Apollo Hospital - Brain & Spine Institute",
            "location": "Bannerghatta Road, Bengaluru, Karnataka",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 80 2630 4050",
            "note": "Specialised neurology department with advanced MRI diagnostics and Alzheimer's management programmes."
        },
        {
            "name": "Fortis Hospital - Neurosciences",
            "location": "Bannerghatta Road, Bengaluru, Karnataka",
            "rating": "4.5 / 5.0",
            "timings": "Mon - Fri: 9:00 AM - 5:00 PM",
            "phone": "+91 80 6621 4444",
            "note": "Offers memory clinics, neuropsychological assessments, and specialised Alzheimer's support groups."
        },
    ],
    "Maharashtra": [
        {
            "name": "Kokilaben Dhirubhai Ambani Hospital - Neurology",
            "location": "Andheri West, Mumbai, Maharashtra",
            "rating": "4.8 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "+91 22 4269 6969",
            "note": "Premier neurocare centre with cutting-edge dementia diagnostics and a dedicated memory clinic."
        },
        {
            "name": "Jaslok Hospital - Department of Neurology",
            "location": "Pedder Road, Mumbai, Maharashtra",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 22 6657 3333",
            "note": "Renowned neurology unit with specialist consultants in cognitive decline and Alzheimer's management."
        },
        {
            "name": "Ruby Hall Clinic - Neurosciences",
            "location": "Sassoon Road, Pune, Maharashtra",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 8:00 AM - 5:00 PM",
            "phone": "+91 20 6645 5555",
            "note": "Leading brain care hospital in Pune offering neurological evaluations and memory disorder treatments."
        },
        {
            "name": "KEM Hospital - Neurology Department",
            "location": "Parel, Mumbai, Maharashtra",
            "rating": "4.5 / 5.0",
            "timings": "Mon - Fri: 8:00 AM - 3:00 PM",
            "phone": "+91 22 2410 7000",
            "note": "Government hospital with a reputed neurology department and affordable Alzheimer's care services."
        },
    ],
    "Tamil Nadu": [
        {
            "name": "Apollo Hospital - Neurosciences",
            "location": "Greams Road, Chennai, Tamil Nadu",
            "rating": "4.8 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "+91 44 2829 0200",
            "note": "One of India's finest neurology units with advanced Alzheimer's research and memory rehabilitation."
        },
        {
            "name": "SIMS Hospital - Neurology",
            "location": "Vadapalani, Chennai, Tamil Nadu",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 44 4392 8888",
            "note": "Shankar Institute of Medical Sciences specialises in neurological disorders and dementia management."
        },
        {
            "name": "Fortis Malar Hospital - Neurology",
            "location": "Adyar, Chennai, Tamil Nadu",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 44 4289 2222",
            "note": "Dedicated brain health unit offering cognitive assessments, MRI diagnostics, and Alzheimer's care."
        },
        {
            "name": "SRMC & RI - Sri Ramachandra Institute",
            "location": "Porur, Chennai, Tamil Nadu",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 8:00 AM - 5:00 PM",
            "phone": "+91 44 4592 8888",
            "note": "Comprehensive neuroscience centre with specialist memory clinics and dementia care programmes."
        },
    ],
    "Delhi": [
        {
            "name": "AIIMS - All India Institute of Medical Sciences",
            "location": "Ansari Nagar, New Delhi",
            "rating": "4.9 / 5.0",
            "timings": "Mon - Sat: 8:00 AM - 4:00 PM",
            "phone": "+91 11 2658 8500",
            "note": "India's most prestigious medical institution with a world-class Neurology & Cognitive Disorders department."
        },
        {
            "name": "Max Super Speciality Hospital - Neurology",
            "location": "Saket, New Delhi",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "+91 11 2651 5050",
            "note": "Advanced neuroscience institute offering memory evaluations, neuroimaging, and Alzheimer's treatment."
        },
        {
            "name": "Fortis Memorial Research Institute - Neurology",
            "location": "Gurugram, Haryana (NCR)",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 124 496 2200",
            "note": "Specialist brain care hospital with a dedicated memory clinic and dementia support services."
        },
        {
            "name": "Sir Ganga Ram Hospital - Neurosciences",
            "location": "Rajinder Nagar, New Delhi",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 11 2575 0000",
            "note": "Renowned for neurological expertise with comprehensive Alzheimer's diagnostics and cognitive rehabilitation."
        },
    ],
    "West Bengal": [
        {
            "name": "AMRI Hospitals - Neurology",
            "location": "Dhakuria, Kolkata, West Bengal",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 33 6606 3800",
            "note": "Advanced Medical Research Institute with a dedicated neurology unit and dementia care services."
        },
        {
            "name": "Apollo Gleneagles Hospital - Neurology",
            "location": "Canal Circular Road, Kolkata, West Bengal",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "+91 33 2320 2020",
            "note": "Specialised neuroscience centre with memory clinics and Alzheimer's management programmes."
        },
        {
            "name": "Medica Superspecialty Hospital - Neurology",
            "location": "Mukundapur, Kolkata, West Bengal",
            "rating": "4.5 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 33 6652 0000",
            "note": "Comprehensive brain health unit offering neurological evaluations and cognitive disorder treatment."
        },
        {
            "name": "Peerless Hospital - Department of Neurology",
            "location": "Panchasayar, Kolkata, West Bengal",
            "rating": "4.5 / 5.0",
            "timings": "Mon - Sat: 8:30 AM - 4:30 PM",
            "phone": "+91 33 4011 1222",
            "note": "Established neurology department with specialist consultants in dementia and memory disorders."
        },
    ],
    "Gujarat": [
        {
            "name": "Sterling Hospital - Neurology",
            "location": "Gurukul Road, Ahmedabad, Gujarat",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 79 4000 4000",
            "note": "Leading multispeciality hospital with a strong neurology division and Alzheimer's support services."
        },
        {
            "name": "Apollo Hospital - Neurosciences",
            "location": "Ahmedabad, Gujarat",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "+91 79 6670 1800",
            "note": "Advanced brain care centre with dedicated memory clinics and neuroimaging diagnostics."
        },
        {
            "name": "CIMS Hospital - Neurology",
            "location": "Science City Road, Ahmedabad, Gujarat",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "+91 79 7126 2000",
            "note": "Care Institute of Medical Sciences with specialist neurologists experienced in dementia and cognitive disorders."
        },
        {
            "name": "HCG Hospital - Brain & Spine",
            "location": "Mithakhali, Ahmedabad, Gujarat",
            "rating": "4.5 / 5.0",
            "timings": "Mon - Fri: 9:00 AM - 5:00 PM",
            "phone": "+91 79 4000 4000",
            "note": "Comprehensive neurology services including cognitive assessments and Alzheimer's management."
        },
    ],
    "Other": [
        {
            "name": "AIIMS - All India Institute of Medical Sciences",
            "location": "Ansari Nagar, New Delhi",
            "rating": "4.9 / 5.0",
            "timings": "Mon - Sat: 8:00 AM - 4:00 PM",
            "phone": "+91 11 2658 8500",
            "note": "India's most prestigious medical institution with a world-class Neurology & Cognitive Disorders department."
        },
        {
            "name": "NIMHANS - National Institute of Mental Health and Neurosciences",
            "location": "Hosur Road, Bengaluru, Karnataka",
            "rating": "4.8 / 5.0",
            "timings": "Mon - Sat: 8:00 AM - 4:00 PM",
            "phone": "+91 80 2699 5000",
            "note": "India's top neuroscience institute with a dedicated Cognitive Disorders Clinic and memory research unit."
        },
        {
            "name": "Apollo Hospitals - Neurology",
            "location": "Multiple locations across India",
            "rating": "4.7 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 6:00 PM",
            "phone": "1860 500 1066 (Apollo Helpline)",
            "note": "Pan-India network of advanced neurology centres with specialist dementia care and memory clinics."
        },
        {
            "name": "Fortis Healthcare - Neurosciences",
            "location": "Multiple locations across India",
            "rating": "4.6 / 5.0",
            "timings": "Mon - Sat: 9:00 AM - 5:00 PM",
            "phone": "1800 103 0201 (Fortis Helpline)",
            "note": "Nationwide chain with dedicated brain health units, memory clinics, and Alzheimer's support services."
        },
    ],
}

INDIA_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Other"
]


# --- 10. PDF GENERATOR ---
def create_pdf(result, confidence, age=None, family_history=None,
               risk_factors=None, cog_score=None, hybrid_label=None,
               hybrid_message=None, orientation=None, memory_recall=None,
               executive_fn=None, cog_test_done=False, image=None, state=None):

    pdf = FPDF()
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(44, 62, 80)
    pdf.rect(0, 0, 210, 32, 'F')
    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(8)
    pdf.cell(0, 16, "ALZ DETECT - Diagnostic Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%B %d, %Y  %H:%M')}", ln=True, align='C')
    pdf.ln(10)

    # ── MRI Analysis Results + Image ─────────────────────────────────────────
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "MRI Analysis Results", ln=True)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    tmp_img_path = None
    if image is not None:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image.save(tmp.name)
            tmp.close()
            tmp_img_path = tmp.name
        except Exception:
            tmp_img_path = None

    img_w = 55
    text_col_w = 190 - img_w - 8
    y_before = pdf.get_y()

    pdf.set_x(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(text_col_w, 9, f"AI Diagnosis:     {result}", ln=True)
    pdf.set_x(10)
    pdf.set_font("Arial", '', 12)
    pdf.cell(text_col_w, 9, f"Confidence:       {confidence:.2f}%", ln=True)

    if tmp_img_path:
        pdf.image(tmp_img_path, x=210 - img_w - 10, y=y_before, w=img_w)
        os.unlink(tmp_img_path)

    img_bottom = y_before + img_w + 2
    if pdf.get_y() < img_bottom:
        pdf.set_y(img_bottom)
    pdf.ln(5)

    # ── Patient Clinical Profile ─────────────────────────────────────────────
    if age is not None:
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, "Patient Clinical Profile", ln=True)
        pdf.set_draw_color(44, 62, 80)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 9, f"Age:                    {age} years", ln=True)
        pdf.cell(0, 9, f"State / Region:         {state if state else 'Not specified'}", ln=True)
        pdf.cell(0, 9, f"Family History:         {family_history}", ln=True)
        active_risks = [r for r, v in (risk_factors or {}).items() if v]
        risk_str = ", ".join(active_risks) if active_risks else "None reported"
        pdf.cell(0, 9, f"Vascular Risk Factors:  {risk_str}", ln=True)
        pdf.ln(5)

    # ── Mini-Cognitive Test Results ──────────────────────────────────────────
    if cog_test_done and cog_score is not None:
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, "Mini-Cognitive Test Results", ln=True)
        pdf.set_draw_color(44, 62, 80)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 9, f"Orientation (Today's date & location):  {orientation}", ln=True)
        pdf.cell(0, 9, f"Memory Recall (3-word delay test):      {memory_recall}", ln=True)
        pdf.cell(0, 9, f"Executive Function (100-7 serial):      {executive_fn}", ln=True)
        pdf.ln(3)
        pdf.set_font("Arial", 'B', 13)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, f"Cognitive Health Score:  {cog_score} / 10", ln=True)
        pdf.ln(5)

    # ── Hybrid Assessment ────────────────────────────────────────────────────
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Hybrid Assessment", ln=True)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    if cog_test_done and hybrid_label and hybrid_message:
        clean_label = hybrid_label.replace("🟢", "").replace("🔴", "").replace("⚠️", "").strip()
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 9, f"Result: {clean_label}", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, hybrid_message)
    else:
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 8, "No cognitive test was completed. Hybrid assessment is unavailable for this report.")
    pdf.ln(5)

    # ── Clinical Recommendations ─────────────────────────────────────────────
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Clinical Recommendations", ln=True)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)

    if result == "Non Demented":
        advice = [
            "1. Engage in regular physical cardiovascular exercise.",
            "2. Maintain a brain-healthy diet (e.g., Mediterranean diet).",
            "3. Stay mentally active with puzzles, reading, or learning new skills.",
            "4. Ensure 7-8 hours of quality sleep nightly to clear brain toxins.",
            "5. Manage cardiovascular risk factors like blood pressure and cholesterol."
        ]
    else:
        advice = [
            "1. CLINICAL REVIEW: Schedule an appointment with a neurologist immediately.",
            "2. ROUTINE: Establish a consistent daily routine to minimize confusion.",
            "3. MEMORY AIDS: Use visible calendars, clocks, and labeled cabinets.",
            "4. SAFETY: Remove tripping hazards and ensure bright home lighting.",
            "5. SUPPORT: Involve family members and explore local caregiver support groups."
        ]

    for line in advice:
        pdf.cell(0, 9, line, ln=True)

    # ── Specialist Hospital Referrals ────────────────────────────────────────
    if result in ["Mild Demented", "Moderate Demented"]:
        pdf.ln(6)
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, "Recommended Specialist Centres", ln=True)
        pdf.set_draw_color(44, 62, 80)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        hospitals    = INDIA_HOSPITALS.get(state, INDIA_HOSPITALS["Other"])
        region_label = state if state and state != "Other" else "India"

        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 6,
            f"The following centres in {region_label} specialise in Alzheimer's disease "
            "and cognitive neurology. Please contact them to schedule a formal evaluation."
        )
        pdf.ln(5)

        for i, h in enumerate(hospitals):
            pdf.set_fill_color(235, 240, 245)
            pdf.set_font("Arial", 'B', 11)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 9, f"  {i+1}. {h['name']}", ln=True, fill=True)
            pdf.ln(2)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(6)
            pdf.cell(0, 7, f"Location :  {h['location']}", ln=True)
            pdf.cell(6)
            pdf.cell(0, 7, f"Rating   :  {h['rating']}", ln=True)
            pdf.cell(6)
            pdf.cell(0, 7, f"Timings  :  {h['timings']}", ln=True)
            pdf.cell(6)
            pdf.cell(0, 7, f"Phone    :  {h['phone']}", ln=True)
            pdf.cell(6)
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 6, f"Note: {h['note']}")
            pdf.ln(4)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(127, 140, 141)
    pdf.multi_cell(0, 6,
        "Disclaimer: ALZ DETECT is an AI-assisted research tool intended to support - not replace - "
        "professional medical diagnosis. All findings must be reviewed by a qualified clinician.",
        align='C'
    )

    return pdf.output(dest='S').encode('latin-1')


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Navigation (always at top) ───────────────────────────────────────────
    st.markdown("## 📋 Dashboard Menu")
    page = st.radio(
        "",
        options=["🏠 Home (MRI Scan)", "📊 Patient Risk Profile"],
        key="nav_page"
    )
    st.divider()

    # ── Patient Clinical Profile ─────────────────────────────────────────────
    st.markdown("## 🏥 Patient Clinical Profile")
    st.markdown("### Demographics & History")

    age = st.slider("Patient Age", min_value=50, max_value=100, value=65, step=1)

    state = st.selectbox(
        "State / Region",
        options=INDIA_STATES,
        index=INDIA_STATES.index("Telangana")
    )

    family_history = st.radio(
        "Family History of Dementia",
        options=["No", "Yes"],
        horizontal=True
    )

    st.markdown("**Vascular Risk Factors**")
    hypertension = st.checkbox("High Blood Pressure")
    diabetes     = st.checkbox("Diabetes")
    smoking      = st.checkbox("Smoking")

    risk_factors = {
        "High Blood Pressure": hypertension,
        "Diabetes":            diabetes,
        "Smoking":             smoking
    }

    st.divider()

    # ── Mini-Cognitive Test ──────────────────────────────────────────────────
    st.markdown("### 🧩 Mini-Cognitive Test")
    st.caption("Answer based on direct patient interaction.")

    orientation = st.radio(
        "**1. Orientation**  \nDoes the patient know today's exact date and their current location?",
        options=["Yes", "No"],
        horizontal=True,
        key="orientation"
    )

    memory_recall = st.radio(
        "**2. Memory Recall**  \nCan the patient repeat 3 specific words after a 5-minute delay?",
        options=["Pass", "Fail"],
        horizontal=True,
        key="memory_recall"
    )

    executive_fn = st.radio(
        "**3. Executive Function**  \nCan the patient count backward from 100 by 7s for five iterations?",
        options=["Pass", "Fail"],
        horizontal=True,
        key="executive_fn"
    )

    cog_test_done = st.checkbox("✅ Cognitive test has been administered", value=False)

    if cog_test_done:
        cog_score = calculate_cognitive_score(orientation, memory_recall, executive_fn)
        risk_label, risk_pct = cognitive_risk_label(cog_score)
        st.divider()
        st.markdown("**Cognitive Health Score:**")
        st.markdown(
            f"<h2 style='color:#ECF0F1; margin:0'>{cog_score} "
            f"<span style='font-size:1rem;color:#BDC3C7'>/ 10</span></h2>",
            unsafe_allow_html=True
        )
        st.progress(cog_score / 10)
        st.caption(f"Cognitive Decline Risk: **{risk_label}**")
    else:
        cog_score = None


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME (MRI SCAN)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home (MRI Scan)":

    st.markdown('<h1 style="text-align:center;font-size:3rem;font-weight:700;color:#2C3E50;margin-bottom:0px;">🧠 ALZ DETECT</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center;font-size:1.3rem;color:#7F8C8D;margin-top:5px;margin-bottom:20px;">Multimodal AI Diagnostic Assistant</h3>', unsafe_allow_html=True)
    st.markdown("""
        <p style="text-align:center;font-size:1.05rem;line-height:1.6;">
            ALZ DETECT combines a Convolutional Neural Network with patient clinical history
            and cognitive testing to deliver a <b>Hybrid Diagnostic Assessment</b>.<br><br>
            <b>Complete the Patient Profile in the sidebar, then upload a grayscale MRI scan.</b>
        </p>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader("Upload Brain MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        col1, col2 = st.columns([1,1])

        with col1:
            st.markdown("**Uploaded Scan:**")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, width=215,)

        with col2:
            st.markdown("**Analysis Results:**")

            with st.spinner("Analyzing scan..."):
                img       = image.resize((176, 176))
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                is_valid, msg = validate_mri(img_array)

                if not is_valid:
                    st.error("🚨 " + msg)
                else:
                    predictions           = model.predict(img_array, verbose=0)
                    predicted_class_index = np.argmax(predictions[0])
                    confidence            = np.max(predictions[0]) * 100
                    class_names           = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
                    result                = class_names[predicted_class_index]

                    # ── Persist to session state ──────────────────────────
                    st.session_state.result           = result
                    st.session_state.confidence       = confidence
                    st.session_state.mri_image        = image
                    st.session_state.age              = age
                    st.session_state.family_history   = family_history
                    st.session_state.cog_score        = cog_score
                    st.session_state.cog_test_done    = cog_test_done
                    st.session_state.risk_factors     = risk_factors
                    st.session_state.state            = state
                    st.session_state.saved_orientation   = orientation
                    st.session_state.saved_memory_recall = memory_recall
                    st.session_state.saved_executive_fn  = executive_fn
                    st.session_state.analysis_complete = True

                    st.success("Analysis Complete!")

                    # ── Dual Metrics ──────────────────────────────────────
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(
                            label="🧠 MRI Atrophy Risk",
                            value=result,
                            delta=f"{confidence:.1f}% confidence"
                        )
                    with m2:
                        if cog_test_done and cog_score is not None:
                            cog_risk, _ = cognitive_risk_label(cog_score)
                            st.metric(
                                label="🧩 Cognitive Decline Risk",
                                value=cog_risk,
                                delta=f"Score: {cog_score}/10"
                            )
                        else:
                            st.metric(
                                label="🧩 Cognitive Decline Risk",
                                value="N/A",
                                delta="Complete sidebar test"
                            )

                    st.progress(int(confidence))
                    st.caption(f"MRI Confidence: {confidence:.2f}%")

                    # ── Hybrid Assessment ─────────────────────────────────
                    st.markdown("---")
                    hybrid_label, hybrid_message, box_class = fusion_engine(
                        result, cog_score if cog_test_done else None,
                        age, family_history, cog_test_done
                    )

                    st.markdown(
                        f"""
                        <div class="hybrid-box {box_class}">
                            <h4>Hybrid Assessment: {hybrid_label}</h4>
                            <p>{hybrid_message}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── PDF Download ──────────────────────────────────────
                    pdf_bytes = create_pdf(
                        result=result,
                        confidence=confidence,
                        age=age,
                        family_history=family_history,
                        risk_factors=risk_factors,
                        cog_score=cog_score if cog_test_done else None,
                        hybrid_label=hybrid_label,
                        hybrid_message=hybrid_message,
                        orientation=st.session_state.saved_orientation if cog_test_done else None,
                        memory_recall=st.session_state.saved_memory_recall if cog_test_done else None,
                        executive_fn=st.session_state.saved_executive_fn if cog_test_done else None,
                        cog_test_done=cog_test_done,
                        image=image,
                        state=state
                    )

                    st.download_button(
                        label="📄 Download Full Diagnostic Report (PDF)",
                        data=pdf_bytes,
                        file_name="ALZ_DETECT_Report.pdf",
                        mime="application/pdf"
                    )

                    st.caption("⚠️ Disclaimer: This is a research tool and should not replace professional medical diagnosis.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PATIENT RISK PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Patient Risk Profile":

    st.markdown('<h1 style="text-align:center;font-size:2.5rem;font-weight:700;color:#2C3E50;margin-bottom:0px;">📊 Patient Risk Profile</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center;font-size:1.1rem;color:#7F8C8D;margin-top:5px;margin-bottom:20px;">Holistic Patient Risk Radar</h3>', unsafe_allow_html=True)
    st.divider()

    # ── Access Control ────────────────────────────────────────────────────────
    if not st.session_state.analysis_complete:
        st.error("🚨 Please upload and analyze a brain MRI scan on the 'Home' page first to view the holistic risk profile.")
        st.stop()

    # ── Pull from session state ───────────────────────────────────────────────
    s_result         = st.session_state.result
    s_confidence     = st.session_state.confidence
    s_age            = st.session_state.age
    s_family_history = st.session_state.family_history
    s_cog_score      = st.session_state.cog_score
    s_cog_test_done  = st.session_state.cog_test_done
    s_risk_factors   = st.session_state.risk_factors

    # ── Radar Chart ───────────────────────────────────────────────────────────
    radar_scores = calculate_radar_scores(
        s_result, s_confidence, s_age,
        s_family_history, s_cog_score, s_risk_factors
    )

    dimensions = list(radar_scores.keys())
    values     = list(radar_scores.values())
    bar_colors = ["#C0392B" if v >= 70 else "#E67E22" if v >= 40 else "#27AE60" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=dimensions,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v} / 100" for v in values],
        textposition='inside',
        textfont=dict(family='Roboto, sans-serif', size=12, color='white'),
        hovertemplate='<b>%{y}</b><br>Risk Score: %{x}/100<extra></extra>',
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor='rgba(44,62,80,0.1)',
            tickfont=dict(size=10, color='#7F8C8D'),
            title=dict(text="Risk Score (0 = Low Risk, 100 = High Risk)",
                       font=dict(size=11, color='#7F8C8D')),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=13, family='Roboto, sans-serif', color='#2C3E50'),
            autorange='reversed',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(236,240,241,0.3)',
        margin=dict(t=20, b=40, l=20, r=80),
        height=320,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Score Breakdown ───────────────────────────────────────────────────────
    st.markdown("#### Risk Dimension Breakdown")
    icons = {
        "Genetics":      "🧬",
        "Vascular":      "🫀",
        "Cognition":     "🧩",
        "Brain Atrophy": "🧠",
        "Age":           "📅"
    }
    summaries = {
        "Genetics":      "Hereditary risk based on family history of dementia. Low = no known history.",
        "Vascular":      "Blood vessel health from BP, diabetes & smoking. Low = no risk factors present.",
        "Cognition":     "Inverted cognitive test score. Low = patient performed well on all 3 tests.",
        "Brain Atrophy": "Physical brain changes detected by the CNN on the MRI scan. Low = no atrophy.",
        "Age":           "Age-related vulnerability on a 50-100yr scale. Low = younger patient (50s).",
    }
    cols = st.columns(5)
    for i, (dim, score) in enumerate(radar_scores.items()):
        color = "#C0392B" if score >= 70 else "#E67E22" if score >= 40 else "#27AE60"
        with cols[i]:
            st.markdown(
                f"""
                <div style="text-align:center; padding:12px 8px; border-radius:10px;
                            background:#f4f6f7; border-top: 4px solid {color};">
                    <div style="font-size:1.5rem;">{icons[dim]}</div>
                    <div style="font-size:0.78rem; color:#7F8C8D; font-weight:600;
                    margin:4px 0; font-family:'Roboto', sans-serif;">{dim}</div>
                    <div style="font-size:1.4rem; font-weight:700; color:{color};
                                font-family:'Roboto', sans-serif;">{score}</div>
                    <div style="font-size:0.7rem; color:#BDC3C7; margin-bottom:8px;">/ 100</div>
                    <div style="font-size:0.68rem; color:#7F8C8D; line-height:1.4;
            font-family:'Roboto', sans-serif;">{summaries[dim]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hybrid Assessment ─────────────────────────────────────────────────────
    st.markdown("---")
    hybrid_label, hybrid_message, box_class = fusion_engine(
        s_result, s_cog_score if s_cog_test_done else None,
        s_age, s_family_history, s_cog_test_done
    )

    st.markdown(
        f"""
        <div class="hybrid-box {box_class}">
            <h4>Hybrid Assessment: {hybrid_label}</h4>
            <p>{hybrid_message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("⚠️ Disclaimer: This is a research tool and should not replace professional medical diagnosis.")