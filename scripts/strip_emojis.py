"""
strip_emojis.py
Removes all emoji characters from UI text in app/streamlit_app.py and README.md.
"""

import re

REPLACEMENTS = {
    'page_icon="🛍️"': 'page_icon=None',
    '<span>🎯</span> ': '',
    '<span>🎯</span>': '',
    '"☀️ Light Mode"': '"Light Mode"',
    '"🌙 Dark Mode"': '"Dark Mode"',
    '"📊 Executive Dashboard"': '"Dashboard"',
    '"🎯 Customer Segmentation"': '"Customer Segmentation"',
    '"📈 Segment Deep-Dive"': '"Segment Analysis"',
    '"🤖 Campaign Prediction"': '"Campaign Prediction"',
    '"👤 Customer 360° Profile"': '"Customer Profile"',
    '"📁 Data Upload & Validation"': '"Data Upload"',
    '"🔬 Methodology & Leakage Audit"': '"Methodology & Audit"',
    '✓ Validated & De-duplicated': 'Validated & De-duplicated',
    '### 🎯 Model 1:': '### Model 1:',
    '#### 📋 Dynamic Segment Profiles': '#### Dynamic Segment Profiles',
    '### 📈 Comprehensive Cross-Segment Behavioral Analysis': '### Cross-Segment Behavioral Analysis',
    '### 🤖 What-If Campaign Simulator: Predict Segment & Response': '### What-If Campaign Simulator: Predict Segment & Response',
    '##### 👤 Demographics': '##### Demographics',
    '##### 🛒 Product Spending ($)': '##### Product Spending ($)',
    '##### 📦 Channels & Engagement': '##### Channels & Engagement',
    '"🔮 Predict Customer Response & Marketing Action"': '"Run Dual-Model Prediction & Generate Strategy"',
    '#### 🎯 Prediction Results & Actionable Decision': '#### Prediction Results & Actionable Decision',
    '### 👤 Customer 360° Profile Dossier': '### Customer 360° Profile Dossier',
    '### 📁 Upload & Validate Customer Dataset': '### Upload & Validate Customer Dataset',
    '✓ All required schema columns are present.': 'All required schema columns are present.',
    '"💾 Apply Cleaned Dataset & Re-Train Pipeline"': '"Apply Cleaned Dataset & Re-Train Pipeline"',
    '### 🔬 Machine Learning Methodology & Data Leakage Prevention': '### Machine Learning Methodology & Data Leakage Prevention',
    '#### 🛡️ Rigorous Data Leakage Prevention Strategy': '#### Data Leakage Prevention Strategy',
    '#### 🏆 Supervised Model Benchmark Comparison': '#### Supervised Model Benchmark Comparison',
    '#### 🧹 Data Quality & Preprocessing Audit Summary': '#### Data Quality & Preprocessing Audit Summary',
    '⚠️ Models or datasets not yet trained.': 'Models or datasets not yet trained.',
    '"🚀 Train Machine Learning Models Now"': '"Train Machine Learning Models Now"'
}

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)

    # Unicode emoji regex range
    emoji_regex = re.compile(
        r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|✓|⚠️',
        flags=re.UNICODE
    )
    cleaned = emoji_regex.sub('', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    remaining = emoji_regex.findall(cleaned)
    print(f"{filepath}: Cleaned! Remaining emojis: {len(remaining)}")

if __name__ == "__main__":
    clean_file("app/streamlit_app.py")
    clean_file("README.md")
