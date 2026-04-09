import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="AIVA - AI Analyst", layout="wide")
st.title("🧠 AIVA - AI Data Analyst (Universal)")

# Upload dataset
uploaded_file = st.file_uploader("data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📊 Dataset Preview")
    st.dataframe(df.head())

    question = st.text_input("💬 Ask a question about your data:")

    if question:

        # ---------- STEP 1: PREPARE CONTEXT ----------
        schema = df.dtypes.to_string()
        sample = df.head(10).to_string()

        # ---------- STEP 2: GPT GENERATES CODE ----------
        prompt = f"""
You are an expert data analyst.

Dataset info:
{schema}

Sample data:
{sample}

The dataset is stored in a pandas DataFrame called df.

A user asked:
"{question}"

Write Python code to answer the question.

Rules:
- Do NOT import anything
- Use only: df, pd, plt
- Store final answer in variable: result
- If useful, generate a chart using matplotlib
- Return ONLY raw Python code (NO markdown, NO ```)

"""

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}]
        )

        code = response.choices[0].message.content

        # ---------- STEP 3: CLEAN CODE ----------
        code = re.sub(r"```.*?\n", "", code)
        code = re.sub(r"```", "", code)
        code = re.sub(r"^import .*", "", code, flags=re.MULTILINE)
        code = re.sub(r"^from .* import .*", "", code, flags=re.MULTILINE)
        code = code.strip()

        st.write("### 🔧 Generated Code")
        st.code(code, language="python")

        # ---------- STEP 4: EXECUTE WITH AUTO-FIX ----------
        local_vars = {"df": df.copy(), "pd": pd, "plt": plt}

        def run_code(code):
            try:
                exec(code, {"__builtins__": {}}, local_vars)
                return local_vars.get("result", None), None
            except Exception as e:
                return None, str(e)

        result, error = run_code(code)

        # ---------- AUTO FIX LOOP ----------
        if error:
            st.warning("⚠️ Fixing code automatically...")

            fix_prompt = f"""
The following Python code failed:

{code}

Error:
{error}

Fix the code.

Rules:
- Do NOT import anything
- Use only df, pd, plt
- Return ONLY corrected Python code
"""

            fix_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": fix_prompt}]
            )

            fixed_code = fix_response.choices[0].message.content

            fixed_code = re.sub(r"```.*?\n", "", fixed_code)
            fixed_code = re.sub(r"```", "", fixed_code)
            fixed_code = fixed_code.strip()

            st.write("### 🛠 Fixed Code")
            st.code(fixed_code, language="python")

            result, error = run_code(fixed_code)

        # ---------- STEP 5: SHOW RESULT ----------
        if error:
            st.error(f"❌ Still failed: {error}")
        else:
            st.write("### 📊 Raw Result")
            st.write(result)

            # ---------- STEP 6: SHOW CHART ----------
            fig = plt.gcf()
            if fig.get_axes():
                st.pyplot(fig)

            # ---------- STEP 7: GPT EXPLAINS LIKE AIVA ----------
            explain_prompt = f"""
You are AIVA, an AI data analyst speaking in a business meeting.

Explain the result clearly and professionally.

Question:
{question}

Result:
{result}
"""

            explain_response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": explain_prompt}]
            )

            explanation = explain_response.choices[0].message.content

            st.write("### 🎤 AIVA Explanation")
            st.success(explanation)
