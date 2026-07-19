import streamlit as st
import pandas as pd
import numpy as np
import joblib
# Page Configuration
st.set_page_config(
    page_title="Lead Scoring Prediction",
    page_icon="📈",
    layout="wide"
)
# Load Model
model = joblib.load("artifacts/model.pkl")
feature_columns = joblib.load("artifacts/feature_columns.pkl")

# Title
st.title("📈 Lead Scoring Prediction System")
st.markdown("""
Predict whether a lead is likely to convert into a customer using the trained XGBoost model.
""")
st.divider()

# Lead Information
st.header("Lead Information")
col1, col2 = st.columns(2)
with col1:
    lead_origin = st.selectbox(
        "Lead Origin",
        [
            "Landing Page Submission",

        "Lead Add Form",

        "Lead Import",

        "Quick Add Form"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Working Professional",
            "Unemployed",
            "Student",
            "Housewife",
            "Other"
        ]
    )
    country = st.selectbox(
    "Country",
    [
        "India",
        "United States",
        "United Kingdom",
        "United Arab Emirates",
        "Australia",
        "Canada",
        "Singapore",
        "South Africa",
        "Saudi Arabia",
        "Qatar",
        "Malaysia",
        "Hong Kong",
        "Germany",
        "France",
        "Italy",
        "China",
        "Bangladesh",
        "Belgium",
        "Bahrain",
        "Denmark",
        "Ghana",
        "Indonesia",
        "Kenya",
        "Kuwait",
        "Liberia",
        "Netherlands",
        "Nigeria",
        "Oman",
        "Philippines",
        "Russia",
        "Sri Lanka",
        "Sweden",
        "Switzerland",
        "Tanzania",
        "Uganda",
        "Vietnam",
        "unknown"
        ]
    )
    specialization = st.selectbox(
    "Specialization",
    [
        "Business Administration",
        "E-Business",
        "E-COMMERCE",
        "Finance Management",
        "Healthcare Management",
        "Hospitality Management",
        "Human Resource Management",
        "IT Projects Management",
        "International Business",
        "Marketing Management",
        "Media and Advertising",
        "Operations Management",
        "Retail Management",
        "Rural and Agribusiness",
        "Services Excellence",
        "Supply Chain Management",
        "Travel and Tourism"
        ]
    )
    city = st.selectbox(
        "City",
        [
        "Other Cities",
        "Other Cities of Maharashtra",
        "Other Metro Cities",
        "Thane & Outskirts",
        "Tier II Cities"
        ]
    )

with col2:
    lead_source = st.selectbox(
        "Lead Source",
        [
        "Direct Traffic",
        "Facebook",
        "Google",
        "Live Chat",
        "NC_EDM",
        "Olark Chat",
        "Organic Search",
        "Pay per Click Ads",
        "Press_Release",
        "Reference",
        "Referral Sites",
        "Social Media",
        "WeLearn",
        "Welingak Website",
        "bing",
        "blog",
        "google",
        "testone",
        "welearnblog_Home",
        "youtubechannel"
        ]
    )

    last_activity = st.selectbox(
        "Last Activity",
        [
         "Converted to Lead",
        "Email Bounced",
        "Email Link Clicked",
        "Email Marked Spam",
        "Email Opened",
        "Email Received",
        "Form Submitted on Website",
        "Had a Phone Conversation",
        "Olark Chat Conversation",
        "Page Visited on Website",
        "Resubscribed to emails",
        "SMS Sent",
        "Unreachable",
        "Unsubscribed",
        "View in browser link Clicked",
        "Visited Booth in Tradeshow"
        ]
    )
    last_notable_activity = st.selectbox(
    "Last Notable Activity",
    [
        "Email Bounced",
        "Email Link Clicked",
        "Email Marked Spam",
        "Email Opened",
        "Email Received",
        "Form Submitted on Website",
        "Had a Phone Conversation",
        "Modified",
        "Olark Chat Conversation",
        "Page Visited on Website",
        "Resubscribed to emails",
        "SMS Sent",
        "Unreachable",
        "Unsubscribed",
        "View in browser link Clicked"
        ]
    )

    lead_quality = st.selectbox(
        "Lead Quality",
        [
        "Low in Relevance",
        "Might be",
        "Not Sure",
        "Worst"
        ]
    )
    tags = st.selectbox(
    "Lead Tag",
    [
        "Busy",
        "Closed by Horizzon",
        "Diploma holder (Not Eligible)",
        "Graduation in progress",
        "In confusion whether part time or DLP",
        "Interested  in full time MBA",
        "Interested in Next batch",
        "Interested in other courses",
        "Lateral student",
        "Lost to EINS",
        "Lost to Others",
        "Not doing further education",
        "Recognition issue (DEC approval)",
        "Ringing",
        "Shall take in the next coming month",
        "Still Thinking",
        "University not recognized",
        "Want to take admission but has financial problems",
        "Will revert after reading the email",
        "in touch with EINS",
        "invalid number",
        "number not provided",
        "opp hangup",
        "switched off",
        "wrong number given"
        ]
    )
st.divider()

# Website Activity
st.header("Website Activity")
col3, col4, col5 = st.columns(3)
with col3:
    total_time_spent = st.number_input(
        "Time Spent on Website",
        min_value=0,
        value=100
    )
with col4:
    total_visits = st.number_input(
        "Total Visits",
        min_value=0,
        value=1
    )
with col5:
    page_views = st.number_input(
        "Page Views Per Visit",
        min_value=0.0,
        value=1.0,
        step=0.1
    )
st.divider()

# Prediction
predict_button = st.button(
    "Predict Lead",
    use_container_width=True
)

if predict_button:
    # Create a feature vector with all training columns
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns
    )
    # Numerical Features
    if "Total Time Spent on Website" in input_data.columns:
        input_data["Total Time Spent on Website"] = total_time_spent

    if "TotalVisits" in input_data.columns:
        input_data["TotalVisits"] = total_visits

    if "Page Views Per Visit" in input_data.columns:
        input_data["Page Views Per Visit"] = page_views
    # One-Hot Encoded Features
    mappings = {
    "Lead Origin": lead_origin,
    "Lead Source": lead_source,
    "What is your current occupation": occupation,
    "Country": country,
    "Specialization": specialization,
    "City": city,
    "Last Activity": last_activity,
    "Last Notable Activity": last_notable_activity,
    "Lead Quality": lead_quality,
    "Tags": tags
    }

    for prefix, value in mappings.items():
        column_name = f"{prefix}_{value}"
        if column_name in input_data.columns:
            input_data[column_name] = 1
    # Prediction
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]
    st.divider()
    st.header("Prediction Result")
    st.metric(
        "Conversion Probability",
        f"{probability*100:.2f}%"
    )

    if prediction == 1:
        st.success("✅ Likely to Convert")
        st.info("""
Recommended Action:
• Prioritize this lead
• Contact within 24 hours
• Assign to Senior Sales Executive
""")
    else:
        st.error("❌ Not Likely to Convert")
        st.warning("""
Recommended Action:
• Nurture through email campaigns
• Follow up after a few days
• Do not prioritize immediate sales effort
""")