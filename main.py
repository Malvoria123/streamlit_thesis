# import modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations
from PIL import Image
import os

st.set_page_config(layout="wide", page_title="CodeGenAI - AI & Programming Insights", page_icon="🤖")

# CSV data path
la_file_id = "1oaTFGnGfUWVwleuROAl2ULYuj4NGRS0W"
DATA_PATH = f"https://drive.google.com/uc?export=download&id={la_file_id}"

OWNERS = [
    {"nim": "2602107650", "name": "Jason Pangestu"},
    {"nim": "2602101823", "name": "Kelvin Alexander Bong"}
]

# column mapping
COL_TEXT = "komen_original"
COL_AI = "Code_Tool"
COL_SENTIMENT = "sentiment"
COL_TIME = "Comment Date"
COL_TASK = "Code_Task"
COL_LANG = "Code_Lang"
COL_PRACFACTOR = "Code_PracFactor"

# create necessary functions

@st.cache_data(ttl=3600)
def load_data(path=DATA_PATH):
    """Load and cache the dataset"""
    try:
        df = pd.read_csv(
            DATA_PATH,
            quotechar='"',
            escapechar='\\',
            encoding='utf-8',
            on_bad_lines='skip'
        )
        return df, None
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except pd.errors.EmptyDataError:
        return None, "CSV file is empty"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"


def split_pipe_list(series):
    """Split pipe-delimited values and deduplicate per row"""
    def parse(x):
        if pd.isna(x) or x == '':
            return []
        if isinstance(x, (list, tuple)):
            items = [str(i).strip() for i in x if str(i).strip() != '']
        else:
            s = str(x)
            parts = [p.strip() for p in s.split('|') if p.strip() != '']
            items = parts

        seen = set()
        out = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out
    return series.apply(parse)


def safe_counter(flat_list):
    """Create counter from flat list, handling empty/invalid values"""
    clean_list = [str(x).strip() for x in flat_list if x and str(x).strip() != '']
    return Counter(clean_list)

# read and preprocess data

df, error = load_data()
if error:
    st.error(error)
    st.stop()

if df is None or df.empty:
    st.error("Dataset is empty or cannot be loaded")
    st.stop()


available_cols = df.columns.tolist()
col_presence = { 
    'text': COL_TEXT in available_cols,
    'ai': COL_AI in available_cols,
    'sentiment': COL_SENTIMENT in available_cols,
    'time': COL_TIME in available_cols,
    'task': COL_TASK in available_cols,
    'lang': COL_LANG in available_cols,
    'pracfactor': COL_PRACFACTOR in available_cols
}

# parse list columns
list_ai = split_pipe_list(df[COL_AI]) if col_presence['ai'] else pd.Series([[] for _ in range(len(df))])
list_task = split_pipe_list(df[COL_TASK]) if col_presence['task'] else pd.Series([[] for _ in range(len(df))])
list_lang = split_pipe_list(df[COL_LANG]) if col_presence['lang'] else pd.Series([[] for _ in range(len(df))])

# pricing flag
pricing_flag = df[COL_PRACFACTOR].astype(str).str.contains('Prac_Pricing_Cost', na=False) if col_presence['pracfactor'] else pd.Series([False]*len(df))

# parse time column
if col_presence['time']:
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors='coerce')

# Custom CSS for larger fonts and better engagement
st.markdown("""
<style>
    /* Increase base font size */
    html, body, [class*="css"] {
        font-size: 16px;
    }
    
    /* Larger headings */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        color: #FAFAFA !important;
    }
    
    h2 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        color: #FAFAFA !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #FAFAFA !important;
    }
    
    /* Larger metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #FAFAFA !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #FAFAFA !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Selectbox and multiselect labels - DARK MODE FRIENDLY */
    .stSelectbox label, .stMultiSelect label {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FAFAFA !important;
        margin-bottom: 0.75rem !important;
        text-shadow: 0 0 10px rgba(255,75,75,0.3);
    }
    
    /* MULTISELECT - ALLOW HORIZONTAL EXPANSION, NO WRAP */
    .stMultiSelect div[data-baseweb="select"] {
        max-width: none !important;
        width: 100% !important;
        min-height: 100px !important;
        height: auto !important;
        overflow-x: auto !important;
        padding: 2rem !important;
    }
    
    .stMultiSelect div[data-baseweb="select"] > div {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        padding: 2rem !important;
        min-height: 100px !important;
        height: auto !important;
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border: 2px solid #FF4B4B !important;
        white-space: nowrap !important;
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        display: flex !important;
        align-items: center !important;
    }
    
    /* SELECTBOX - Keep wrapping for long queries */
    .stSelectbox div[data-baseweb="select"] {
        max-width: 100% !important;
        width: 100% !important;
        min-height: 60px !important;
        height: auto !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        min-height: 60px !important;
        height: auto !important;
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border: 2px solid #FF4B4B !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        line-height: 1.5 !important;
    }
    
    /* CRITICAL: Fix selected text visibility - ALL ELEMENTS */
    .stSelectbox input,
    .stSelectbox div[data-baseweb="select"] input,
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] div[role="button"],
    .stSelectbox [data-baseweb="select"] [class*="singleValue"],
    .stSelectbox [data-baseweb="select"] [class*="placeholder"],
    .stSelectbox [data-baseweb="select"] > div > div {
        color: #FAFAFA !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        line-height: 1.5 !important;
    }
    
    .stMultiSelect input,
    .stMultiSelect div[data-baseweb="select"] input,
    .stMultiSelect div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] div {
        color: #FAFAFA !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }
    
    /* Dropdown menu items - MUCH BIGGER */
    [data-baseweb="menu"] {
        background-color: #262730 !important;
        max-width: none !important;
    }
    
    [data-baseweb="menu"] li {
        font-size: 1.8rem !important;
        padding: 1rem 1.5rem !important;
        color: #FAFAFA !important;
        font-weight: 500 !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.5 !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    
    /* Selected tags in multiselect - BIGGER AND NO TRUNCATION */
    .stMultiSelect span[data-baseweb="tag"] {
        font-size: 1.6rem !important;
        padding: 0.75rem 1.25rem !important;
        background-color: #FF4B4B !important;
        color: white !important;
        font-weight: 600 !important;
        margin: 0.25rem !important;
        max-width: none !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    
    /* Tag text inside */
    .stMultiSelect span[data-baseweb="tag"] span {
        max-width: none !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
    }
    
    /* MUCH LARGER expander */
    .streamlit-expanderHeader {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        padding: 1.25rem 1.5rem !important;
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border: 2px solid #FF4B4B !important;
        border-radius: 8px !important;
    }
    
    /* Larger buttons - ESPECIALLY DOWNLOAD */
    .stButton > button, .stDownloadButton > button {
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        background-color: #FF4B4B !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #FF6B6B !important;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(255,75,75,0.4);
    }
    
    /* ========================================= */
    /* DATAFRAME - TARGETED FIX FOR VISIBILITY */
    /* ========================================= */
    
    /* Container backgrounds */
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] > div {
        background-color: transparent !important;
    }
    
    /* Table styling */
    div[data-testid="stDataFrame"] table,
    .dataframe {
        background-color: #262730 !important;
        border-collapse: collapse !important;
        width: 100% !important;
    }
    
    /* Header cells */
    div[data-testid="stDataFrame"] table thead th,
    .dataframe thead th {
        background-color: #1a1d24 !important;
        color: #FAFAFA !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem !important;
        border: 1px solid #3a3d44 !important;
        text-align: left !important;
    }
    
    /* Body cells */
    div[data-testid="stDataFrame"] table tbody td,
    .dataframe tbody td {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        font-size: 1rem !important;
        padding: 0.5rem 0.75rem !important;
        border: 1px solid #3a3d44 !important;
    }
    
    /* Index column */
    div[data-testid="stDataFrame"] table tbody th,
    .dataframe tbody th {
        background-color: #1a1d24 !important;
        color: #FAFAFA !important;
        font-weight: 600 !important;
        padding: 0.5rem 0.75rem !important;
        border: 1px solid #3a3d44 !important;
    }
    
    /* Hover effect */
    div[data-testid="stDataFrame"] table tbody tr:hover td,
    .dataframe tbody tr:hover td {
        background-color: #2a2d34 !important;
    }
    
    /* Ensure text is visible in all table elements */
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td,
    .dataframe th,
    .dataframe td {
        color: #FAFAFA !important;
    }
    
    /* Pandas HTML table styling */
    table.dataframe,
    .stMarkdown table {
        background-color: #262730 !important;
        border-collapse: collapse !important;
        width: 100% !important;
        margin: 1rem 0 !important;
    }
    
    .stMarkdown table thead th,
    table.dataframe thead th {
        background-color: #1a1d24 !important;
        color: #FAFAFA !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem !important;
        border: 1px solid #3a3d44 !important;
        text-align: left !important;
    }
    
    .stMarkdown table tbody td,
    table.dataframe tbody td {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        font-size: 1rem !important;
        padding: 0.5rem 0.75rem !important;
        border: 1px solid #3a3d44 !important;
    }
    
    .stMarkdown table tbody th,
    table.dataframe tbody th {
        background-color: #1a1d24 !important;
        color: #FAFAFA !important;
        font-weight: 600 !important;
        padding: 0.5rem 0.75rem !important;
        border: 1px solid #3a3d44 !important;
        text-align: right !important;
    }
    
    .stMarkdown table tbody tr:hover td,
    table.dataframe tbody tr:hover td {
        background-color: #2a2d34 !important;
    }
    
    /* Highlight sections */
    .highlight-box {
        background-color: #1a1d24;
        padding: 2.5rem;
        border-radius: 12px;
        border-left: 6px solid #FF4B4B;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .insight-box {
        background-color: #1a2332;
        padding: 2.5rem;
        border-radius: 12px;
        border-left: 6px solid #2196F3;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(33,150,243,0.2);
    }
    
    .insight-box h3 {
        color: #64B5F6 !important;
        font-size: 2.2rem !important;
        margin-bottom: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    .insight-box p {
        font-size: 2rem !important;
        line-height: 2 !important;
        color: #E3F2FD !important;
        font-weight: 500 !important;
        background-color: rgba(33,150,243,0.1);
        padding: 1.5rem;
        border-radius: 8px;
    }
    
    .stats-box {
        background-color: #2a2416;
        padding: 2.5rem;
        border-radius: 12px;
        border-left: 6px solid #FF9800;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255,152,0,0.2);
    }
    
    .stats-box h3 {
        color: #FFB74D !important;
        font-size: 1.8rem !important;
        margin-bottom: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    .about-section {
        background-color: #1a1d24;
        padding: 2.5rem;
        border-radius: 15px;
        margin-top: 3rem;
        border: 2px solid #FF4B4B;
        color: #FAFAFA !important;
    }
    
    /* Larger paragraph text */
    p {
        font-size: 1.2rem !important;
        line-height: 1.8 !important;
        color: #FAFAFA !important;
    }
    
    /* BOLD text inside paragraphs - VERY PROMINENT */
    p strong, p b, strong, b {
        font-size: 1.5rem !important;
        font-weight: 900 !important;
        color: #FF6B6B !important;
        text-shadow: 0 0 10px rgba(255,107,107,0.3);
    }
    
    /* Markdown bold in insight box - EVEN BIGGER */
    .insight-box p strong, .insight-box p b {
        color: #64B5F6 !important;
        font-size: 2.4rem !important;
        font-weight: 900 !important;
    }
    
    /* Info boxes */
    .stAlert {
        font-size: 1.3rem !important;
        background-color: #262730 !important;
        color: #FAFAFA !important;
    }
    
    /* Remove ALL empty blocks and spaces - AGGRESSIVE */
    div[data-testid="stVerticalBlock"] > div:empty,
    div[data-testid="stHorizontalBlock"] > div:empty,
    div.element-container:empty,
    div[class*="css"]:empty,
    .stMarkdown:empty,
    p:empty,
    div:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }
    
    /* Remove extra spacing */
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Ensure no white blocks in dark mode */
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
    }
    
    /* Remove gaps between elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to convert markdown bold to HTML
def markdown_to_html_bold(text):
    """Convert **text** to <strong>text</strong> for better rendering"""
    import re
    # Replace **text** with <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return text

# Streamlit UI

# Title with logo
logo_col, title_col = st.columns([1, 9])
with logo_col:
    if os.path.exists("logo.png"):
        try:
            logo = Image.open("logo.png")
            st.image(logo, width=100)
        except:
            st.write("🤖")
    else:
        st.write("🤖")

with title_col:
    st.title("CodeGenAI")
    st.markdown("### An Interactive Application for Analysis on Generative AI Based Programming Tools")
    st.markdown("#### 🔗 The source of the data is scraped from Reddit platform inside forums that talk about AI & Programming")

st.markdown("---")

# ========== STATISTIK DATA (TOP SECTION) ==========
st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
st.markdown("## � DATA STATISTICS")

# Calculate statistics
total_ai_mentions = sum(len(x) for x in list_ai) if col_presence['ai'] else 0

# Display metrics in columns
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("📝 Number of Rows", f"{len(df):,}")
with metric_col2:
    st.metric("📋 Total Columns", f"{len(available_cols)}")
with metric_col3:
    if col_presence['ai']:
        st.metric("🤖 Total AI Mentions", f"{total_ai_mentions:,}")
    else:
        st.metric("🤖 Total AI Mentions", "N/A")

st.markdown('</div>', unsafe_allow_html=True)

# data preview
with st.expander("👀 Dataset Preview (10 First Rows)", expanded=False):
    if col_presence['text']:
        preview_df = df[[COL_TEXT]].head(10).copy()
        preview_df.index = range(1, len(preview_df) + 1)
        
        # Use HTML table for better dark mode visibility
        st.markdown(preview_df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.warning(f"Column '{COL_TEXT}' not found")
        st.write("Available columns:", available_cols)

st.markdown("---")

# ========== FILTER ANALISIS (MIDDLE SECTION) ==========
st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
st.markdown("## 🔍 ANALYSIS FILTER")

# section selection
section_options = {
    "🤖 AI Usage": [
        ("Q1", "What are the most used AI tools?"),
        ("Q2", "What is the proportion of each AI tool usage (%)?"),
        ("Q3", "What is the sentiment distribution of each AI?"),
        ("Q4", "What programming tasks are associated with AI tools?"),
        ("Q5", "What is the trend of AI usage from time to time?"),
        ("Q6", "What AI tool is associated with Pricing/Cost?")
    ],
    "💻 Programming Language": [
        ("Q7", "What programming languages are the most popular?"),
        ("Q8", "What programming languages are associated with each AI tool?"),
        ("Q9", "What AI tools are dominant in each programming language?")
    ],
    "📋 Programming Task": [
        ("Q10", "What programming tasks are the most talked about?"),
        ("Q11", "What programming tasks are associated with specific AI tool?")
    ]
}

selected_sections = st.multiselect(
    "Choose Analysis Category:",
    options=list(section_options.keys()),
    default=list(section_options.keys()),
    help="Select one or more categories to filter available queries"
)

st.markdown('</div>', unsafe_allow_html=True)

# build query selection
available_queries = []
for sec in selected_sections:
    available_queries.extend(section_options[sec])

if not available_queries:
    st.warning("⚠️ Please choose at least one category to continue")
    st.stop()

query_label_map = {q[0]: q[1] for q in available_queries}
query_keys = [q[0] for q in available_queries]

# create analysis functions for each query

def answer_q1(df):
    """Most frequently used AI tools"""
    if not col_presence['ai']:
        return None, None, "Column Code_Tool not found."
    
    flat = [it for row in list_ai for it in row]
    if not flat:
        return None, None, "No AI tools data detected."
    
    c = safe_counter(flat)
    total_mentions = sum(c.values())
    
    # COMPLETE TABLE - all data
    table_complete = pd.DataFrame(c.most_common(), columns=['AI Tool', 'Count'])
    
    # Chart only top 20
    table_chart = table_complete.head(20)
    fig = px.bar(
        table_chart, 
        x='AI Tool', 
        y='Count',
        title='Top 20 Most Frequently Used AI Tools',
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    top_tool = table_complete.iloc[0]['AI Tool']
    top_count = table_complete.iloc[0]['Count']
    top_pct = (top_count / total_mentions * 100) if total_mentions > 0 else 0
    
    text = f"**{top_tool}** is the most frequently used AI tool with **{top_count:,} mentions** ({top_pct:.1f}% of total {total_mentions:,} mentions)."
    
    return fig, table_complete, text


def answer_q2(df):
    """Proportion of AI tool usage"""
    if not col_presence['ai']:
        return None, None, "Column Code_Tool not found."
    
    flat = [it for row in list_ai for it in row]
    if not flat:
        return None, None, "No AI tools data available."
    
    c = safe_counter(flat)
    total = sum(c.values()) or 1
    
    # COMPLETE TABLE - all data
    table_complete = pd.DataFrame(
        [(k, v, v/total*100) for k, v in c.most_common()],
        columns=['AI Tool', 'Count', 'Percentage (%)']
    )
    
    # Chart only top 15
    fig = px.pie(
        table_complete.head(15),
        names='AI Tool',
        values='Percentage (%)',
        title='AI Tools Usage Distribution (Top 15)',
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    text = f"Showing usage proportion of **{len(c)} different AI tools**. Total mentions: **{total:,}**."
    
    return fig, table_complete, text


def answer_q3(df):
    """AI with sentiment analysis"""
    if not (col_presence['ai'] and col_presence['sentiment']):
        return None, None, "Column Code_Tool or sentiment not found."
    
    def label_sent(x):
        if pd.isna(x):
            return 'neutral'
        if isinstance(x, (int, float)):
            if x > 0: return 'positive'
            if x < 0: return 'negative'
            return 'neutral'
        sx = str(x).lower()
        if 'pos' in sx: return 'positive'
        if 'neg' in sx: return 'negative'
        return 'neutral'
    
    sent = df[COL_SENTIMENT].apply(label_sent)
    
    rows = []
    for s, ais in zip(sent, list_ai):
        for a in ais:
            rows.append((a, s))
    
    if not rows:
        return None, None, "No data available for sentiment analysis."
    
    adf = pd.DataFrame(rows, columns=['AI Tool', 'Sentiment'])
    grouping = adf.groupby(['AI Tool', 'Sentiment']).size().reset_index(name='Count')
    
    # COMPLETE TABLE - all data
    pivot_complete = grouping.pivot(index='AI Tool', columns='Sentiment', values='Count').fillna(0)
    pivot_complete['Total'] = pivot_complete.sum(axis=1)
    pivot_complete = pivot_complete.sort_values(by='Total', ascending=False)
    table_complete = pivot_complete.reset_index()
    
    # Chart only top 20
    pivot_chart = pivot_complete.head(20)
    fig = go.Figure()
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
    
    for col in ['positive', 'negative', 'neutral']:
        if col in pivot_chart.columns:
            fig.add_trace(go.Bar(
                name=col.capitalize(),
                x=pivot_chart.index,
                y=pivot_chart[col],
                marker_color=colors.get(col, '#3498db')
            ))
    
    fig.update_layout(
        barmode='stack',
        title='Sentiment Distribution for Top 20 AI Tools',
        xaxis_title='AI Tool',
        yaxis_title='Number of Mentions',
        xaxis_tickangle=-45
    )
    
    text = "Sentiment analysis shows how AI tools are discussed in positive, negative, or neutral contexts."
    
    return fig, table_complete, text


def answer_q4(df):
    """AI for various tasks"""
    if not (col_presence['ai'] and col_presence['task']):
        return None, None, "Column Code_Tool or Code_Task not found."
    
    ctr = Counter()
    for ais, tasks in zip(list_ai, list_task):
        for a in ais:
            for t in tasks:
                if a and t:
                    ctr[(a, t)] += 1
    
    if not ctr:
        return None, None, "No AI-task data found."
    
    # COMPLETE TABLE - all data
    table_complete = pd.DataFrame(
        [(a, t, c) for ((a, t), c) in ctr.most_common()],
        columns=['AI Tool', 'Task', 'Count']
    )
    
    # Chart only top 8 AI and top 50 combinations
    top_ai = table_complete['AI Tool'].value_counts().head(8).index.tolist()
    dfchart = table_complete[table_complete['AI Tool'].isin(top_ai)].head(50)
    
    fig = px.bar(
        dfchart,
        x='AI Tool',
        y='Count',
        color='Task',
        title=f'Tasks Associated with Top {len(top_ai)} AI Tools',
        barmode='stack'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    text = f"Showing **{len(ctr)} unique combinations** of AI-task from dataset."
    
    return fig, table_complete, text


def answer_q5(df):
    """AI usage trends over time"""
    if not (col_presence['ai'] and col_presence['time']):
        return None, None, "Column Comment Date or Code_Tool not available."
    
    rows = []
    for ts, ais in zip(df[COL_TIME], list_ai):
        if pd.notna(ts):
            for a in ais:
                rows.append((ts, a))
    
    if not rows:
        return None, None, "No valid timestamp data available."
    
    tdf = pd.DataFrame(rows, columns=['timestamp', 'AI Tool'])
    tdf['month'] = pd.to_datetime(tdf['timestamp']).dt.to_period('M').dt.to_timestamp()
    
    # COMPLETE TABLE - all data (all AI, all months)
    counts_complete = tdf.groupby(['month', 'AI Tool']).size().reset_index(name='Count')
    
    # Chart only top 6 AI
    top_ai = counts_complete.groupby('AI Tool')['Count'].sum().nlargest(6).index.tolist()
    counts_chart = counts_complete[counts_complete['AI Tool'].isin(top_ai)]
    
    fig = px.line(
        counts_chart,
        x='month',
        y='Count',
        color='AI Tool',
        title=f'Usage Trend of Top {len(top_ai)} AI Tools (Per Month)',
        markers=True
    )
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Mentions')
    
    # Table for details: pivot all data
    table_complete = counts_complete.pivot(index='month', columns='AI Tool', values='Count').fillna(0).reset_index()
    
    text = f"Time-series analysis shows AI usage trends from **{tdf['month'].min().strftime('%B %Y')}** to **{tdf['month'].max().strftime('%B %Y')}**."
    
    return fig, table_complete, text


def answer_q6(df):
    """AI in pricing context"""
    if not col_presence['ai']:
        return None, None, "Column Code_Tool not found."
    
    rows_with_pricing = df[pricing_flag]
    if rows_with_pricing.empty:
        return None, None, "No pricing/cost discussion found in dataset."
    
    rows_ai = split_pipe_list(rows_with_pricing[COL_AI]) if COL_AI in rows_with_pricing.columns else []
    flat = [it for row in rows_ai for it in row]
    
    if not flat:
        return None, None, "No AI tools mentioned in pricing context."
    
    c = safe_counter(flat)
    
    # COMPLETE TABLE - all data
    table_complete = pd.DataFrame(c.most_common(), columns=['AI Tool', 'Count'])
    
    # Chart only top 20
    fig = px.bar(
        table_complete.head(20),
        x='AI Tool',
        y='Count',
        title='AI Tools Discussed in Pricing/Cost Context (Top 20)',
        color='Count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    text = f"From **{len(rows_with_pricing)} discussions** about pricing/cost, found **{len(c)} AI tools** mentioned."
    
    return fig, table_complete, text


def answer_q7(df):
    """Most popular programming languages"""
    if not col_presence['lang']:
        return None, None, "Column Code_Lang not found."
    
    flat = [it for row in list_lang for it in row]
    if not flat:
        return None, None, "No programming language data available."
    
    c = safe_counter(flat)
    total = sum(c.values())
    
    # COMPLETE TABLE - all data
    table_complete = pd.DataFrame(c.most_common(), columns=['Language', 'Count'])
    
    # Chart only top 20
    fig = px.bar(
        table_complete.head(20),
        x='Language',
        y='Count',
        title='Top 20 Most Frequently Discussed Programming Languages',
        color='Count',
        color_continuous_scale='Greens'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    top_lang = table_complete.iloc[0]['Language']
    top_count = table_complete.iloc[0]['Count']
    
    text = f"**{top_lang}** is the most popular language with **{top_count:,} mentions** out of total **{total:,} mentions**."
    
    return fig, table_complete, text


def answer_q8(df):
    """Languages for specific AI"""
    if not (col_presence['ai'] and col_presence['lang']):
        return None, None, "Column Code_Tool or Code_Lang not found."
    
    ctr = Counter()
    for ais, langs in zip(list_ai, list_lang):
        for a in ais:
            for l in langs:
                if a and l:
                    ctr[(a, l)] += 1
    
    if not ctr:
        return None, None, "No AI-language relationship data available."
    
    ai_choices = sorted({k[0] for k in ctr.keys()})
    
    st.markdown("### 🤖 Select AI Tool to Analyze:")
    ai_choice = st.selectbox(
        "Choose AI Tool:",
        options=ai_choices,
        help="See programming languages frequently used with this AI"
    )
    
    pairs = [(lang, cnt) for (a, lang), cnt in ctr.items() if a == ai_choice]
    
    # COMPLETE TABLE - all data for selected AI
    table_complete = pd.DataFrame(sorted(pairs, key=lambda x: -x[1]), columns=['Language', 'Count'])
    
    # Chart only top 15
    fig = px.bar(
        table_complete.head(15),
        x='Language',
        y='Count',
        title=f'Most Frequently Used Languages with {ai_choice} (Top 15)',
        color='Count',
        color_continuous_scale='Purples'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    total_mentions = table_complete['Count'].sum()
    text = f"**{ai_choice}** is used with **{len(table_complete)} different languages**, total **{total_mentions:,} mentions**."
    
    return fig, table_complete, text


def answer_q9(df):
    """Dominant AI for specific language"""
    if not (col_presence['ai'] and col_presence['lang']):
        return None, None, "Column Code_Tool or Code_Lang not found."
    
    ctr = Counter()
    for ais, langs in zip(list_ai, list_lang):
        for a in ais:
            for l in langs:
                if a and l:
                    ctr[(l, a)] += 1
    
    if not ctr:
        return None, None, "No language-AI relationship data available."
    
    lang_choices = sorted({k[0] for k in ctr.keys()})
    
    st.markdown("### 💻 Select Programming Language to Analyze:")
    lang_choice = st.selectbox(
        "Choose Programming Language:",
        options=lang_choices,
        help="See AI tools dominant for this language"
    )
    
    pairs = [(ai, cnt) for (lang, ai), cnt in ctr.items() if lang == lang_choice]
    
    # COMPLETE TABLE - all data for selected language
    table_complete = pd.DataFrame(sorted(pairs, key=lambda x: -x[1]), columns=['AI Tool', 'Count'])
    
    # Chart only top 15
    fig = px.bar(
        table_complete.head(15),
        x='AI Tool',
        y='Count',
        title=f'Most Frequently Used AI Tools for {lang_choice} (Top 15)',
        color='Count',
        color_continuous_scale='Oranges'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    total_mentions = table_complete['Count'].sum()
    text = f"For language **{lang_choice}**, found **{len(table_complete)} AI tools** with total **{total_mentions:,} mentions**."
    
    return fig, table_complete, text


def answer_q10(df):
    """Most frequently discussed programming tasks"""
    if not col_presence['task']:
        return None, None, "Column Code_Task not found."
    
    flat = [it for row in list_task for it in row]
    if not flat:
        return None, None, "No task data available."
    
    c = safe_counter(flat)
    total = sum(c.values())
    
    # COMPLETE TABLE - all data
    table_complete = pd.DataFrame(c.most_common(), columns=['Task', 'Count'])
    
    # Chart only top 20
    fig = px.bar(
        table_complete.head(20),
        x='Task',
        y='Count',
        title='Top 20 Most Frequently Discussed Programming Tasks',
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    top_task = table_complete.iloc[0]['Task']
    top_count = table_complete.iloc[0]['Count']
    
    text = f"**{top_task}** is the most popular task with **{top_count:,} mentions** out of total **{total:,} mentions**."
    
    return fig, table_complete, text


def answer_q11(df):
    """Tasks associated with specific AI"""
    if not (col_presence['ai'] and col_presence['task']):
        return None, None, "Column Code_Tool or Code_Task not found."
    
    ctr = Counter()
    for ais, tasks in zip(list_ai, list_task):
        for a in ais:
            for t in tasks:
                if a and t:
                    ctr[(a, t)] += 1
    
    if not ctr:
        return None, None, "No AI-task relationship data available."
    
    ai_choices = sorted({k[0] for k in ctr.keys()})
    
    st.markdown("### 🤖 Select AI Tool to Analyze:")
    ai_choice = st.selectbox(
        "Choose AI Tool:",
        options=ai_choices,
        key='q11_ai',
        help="See tasks most frequently done with this AI"
    )
    
    pairs = [(task, cnt) for (a, task), cnt in ctr.items() if a == ai_choice]
    
    # COMPLETE TABLE - all data for selected AI
    table_complete = pd.DataFrame(sorted(pairs, key=lambda x: -x[1]), columns=['Task', 'Count'])
    
    # Chart only top 20
    fig = px.bar(
        table_complete.head(20),
        x='Task',
        y='Count',
        title=f'Tasks Most Frequently Associated with {ai_choice} (Top 20)',
        color='Count',
        color_continuous_scale='RdYlBu'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    total_tasks = table_complete['Count'].sum()
    text = f"**{ai_choice}** is used for **{len(table_complete)} types of tasks** with total **{total_tasks:,} mentions**."
    
    return fig, table_complete, text


# function mapping
answer_map = {
    'Q1': answer_q1, 'Q2': answer_q2, 'Q3': answer_q3, 'Q4': answer_q4,
    'Q5': answer_q5, 'Q6': answer_q6, 'Q7': answer_q7, 'Q8': answer_q8,
    'Q9': answer_q9, 'Q10': answer_q10, 'Q11': answer_q11
}


# ========== FILTER QUERY (QUERY SELECTION) ==========
st.markdown("---")
st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
st.markdown("## 📋 CHOOSE QUERY")

query_choice = st.selectbox(
    "Choose Analysis Query:",
    options=[(k + ' - ' + query_label_map[k]) for k in query_keys],
    help="Choose the analysis question to be answered"
)

selected_q_key = query_choice.split(' - ')[0]
st.markdown('</div>', unsafe_allow_html=True)



# perform analysis
with st.spinner("🔄 Processing analysis..."):
    func = answer_map.get(selected_q_key)
    if func is None:
        st.error("Query not yet implemented.")
    else:
        try:
            chart, table, explanation = func(df)
            
            
            st.markdown("---")
            st.markdown(f"## 📊 Analysis Results: {query_label_map[selected_q_key]}")
            
            # CHART FIRST - Full width
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("📉 Chart not available: " + (explanation or "No data available."))
            
            # INSIGHT AND QUICK STATS BELOW - Side by side with highlights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 💡 Insight")
                # Convert markdown bold to HTML for better rendering
                explanation_html = markdown_to_html_bold(explanation)
                st.markdown(f"<p style='font-size: 2rem; line-height: 2; color: #E3F2FD;'>{explanation_html}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if table is not None and not table.empty:
                    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                    st.markdown("### 📈 Quick Stats")
                    
                    # Total Items = number of unique items in ALL data
                    st.metric("📊 Total Items", f"{len(table):,}", 
                             help="Number of unique entities found in dataset")
                    
                    # Total Mentions = sum of 'Count' column for ALL data
                    if 'Count' in table.columns:
                        total_mentions = table['Count'].sum()
                        st.metric("🔢 Total Mentions", f"{total_mentions:,}",
                                 help="Total occurrences of all items (including duplicates)")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # detailed table - SHOWING ALL DATA
            if table is not None and not table.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"### 📋 VIEW DETAILED DATA")
                with st.expander(f"Click to expand — Total: {len(table):,} items", expanded=False):
                    # Use HTML table for better dark mode visibility
                    st.markdown(
                        f'<div style="max-height: 400px; overflow-y: auto;">{table.to_html(escape=False, index=False)}</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # download option - MUCH MORE PROMINENT
                    csv = table.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ DOWNLOAD COMPLETE CSV FILE",
                        data=csv,
                        file_name=f"analysis_{selected_q_key}_complete.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("Table not available: " + (explanation or "No data available."))
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            with st.expander("🐛 Error Details"):
                st.exception(e)


# ========== ABOUT US (BOTTOM SECTION) ==========
st.markdown("---")
st.markdown('<div class="about-section">', unsafe_allow_html=True)
st.markdown("## 👥 ABOUT US")

st.markdown("""
This application is developed as part of a **Non-Class Thesis** project at **Bina Nusantara University**. 
The project focuses on analyzing the usage patterns and trends of Generative AI-based programming tools through 
data scraped from Reddit forums dedicated to AI and programming discussions.

**Team Members:**

**1. Jason Pangestu (NIM: 2602107650)**  
Highly responsible in coding, creating application, and strategizing the thesis process. Jason led the technical 
development of this interactive dashboard, implementing data processing pipelines, visualization components, and 
ensuring the application's deployment readiness.

**2. Kelvin Alexander Bong (NIM: 2602101823)**  
Highly responsible in writing, creating poster, and discussing with the supervisor. Kelvin managed the research 
documentation, academic writing, visual presentation materials, and coordinated with the thesis supervisor to 
ensure the project meets academic standards.

---

**Project Objectives:**  
This research aims to understand how developers discuss and use AI powered coding tools, which programming languages and tasks are most closely associated with these tools, and how sentiment differs across AI platforms. The study also explicitly adopts a Computational Grounded Theory approach to move beyond surface level patterns and uncover how AI is experienced, evaluated, and adapted within everyday programming practices. Through this approach, the findings contribute to a deeper understanding of AI assisted software development as an evolving and context dependent process.

**Data Source:**  
All data presented in this application is sourced from public Reddit discussions within communities focused on AI and programming topics. The dataset was collected using Reddit's official API, with a maximum of 40 posts per keyword search and up to 4,000 comments per post, applying a delay of 1.5 seconds per request to comply with platform guidelines. Only popular posts with high engagement were included, and all data was restricted to content published after January 1, 2022 to ensure relevance to contemporary AI coding tools.

**Research Process:**  
The research process begins with automated data scraping, followed by data cleaning and exploratory data analysis to understand overall patterns. Topic modeling is then applied as an exploratory step to surface dominant discussion themes, which are further examined using Grounded Theory coding to identify meaningful categories and relationships. Sentiment analysis is integrated to capture evaluative perspectives, and the results are systematically reviewed and evaluated to ensure that the identified patterns accurately reflect developer discourse and usage contexts.
""")

# Flow diagram - separate markdown with HTML
st.markdown("""
<div style="background-color: #1a1d24; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border: 2px solid #2196F3;">
    <div style="text-align: center; font-size: 0.95rem; line-height: 1.5;">
        <div style="margin: 0.5rem 0;">
            <span style="background-color: #2196F3; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; display: inline-block;">
                📥 Data Scraping
            </span>
        </div>
        <div style="color: #64B5F6; font-size: 1.2rem; margin: 0.25rem 0;">↓</div>
        <div style="margin: 0.5rem 0;">
            <span style="background-color: #2196F3; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; display: inline-block;">
                🧹 Data Cleaning & EDA
            </span>
        </div>
        <div style="color: #64B5F6; font-size: 1.2rem; margin: 0.25rem 0;">↓</div>
        <div style="margin: 0.5rem 0;">
            <span style="background-color: #2196F3; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; display: inline-block;">
                🔍 Topic Modeling
            </span>
        </div>
        <div style="color: #64B5F6; font-size: 1.2rem; margin: 0.25rem 0;">↓</div>
        <div style="margin: 0.5rem 0;">
            <span style="background-color: #2196F3; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; display: inline-block;">
                📝 Grounded Theory
            </span>
        </div>
        <div style="color: #64B5F6; font-size: 1.2rem; margin: 0.25rem 0;">↓</div>
        <div style="margin: 0.5rem 0;">
            <span style="background-color: #2196F3; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; display: inline-block;">
                😊 Sentiment Analysis
            </span>
        </div>
        <div style="color: #64B5F6; font-size: 1.2rem; margin: 0.25rem 0;">↓</div>
        <div style="margin: 0.5rem 0;">
            <span style="background-color: #FF9800; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; display: inline-block;">
                ✅ Review & Evaluation
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ========== RESULT SECTION ==========
st.markdown("---")
st.markdown("## 🎯 RESULT")

# Result text only - no diagram
st.markdown("""
This study shows that developer discussions on AI coding tools follow a recurring cycle. AI is used to support specific programming tasks, the resulting outputs are evaluated based on technical and practical considerations, and these evaluations lead to contextual adaptations in how AI is used. The core category, **Tool-Mediated Adaptive Programming Practice**, highlights AI as a working infrastructure that shapes developers' tasks and decisions over time.
""")

# footer
st.markdown("---")
missing_cols = [k for k, v in col_presence.items() if not v]
if missing_cols:
    with st.expander("⚠️ Warning: Some columns not found"):
        col_names = {
            'text': COL_TEXT, 'ai': COL_AI, 'sentiment': COL_SENTIMENT,
            'time': COL_TIME, 'task': COL_TASK, 'lang': COL_LANG, 'pracfactor': COL_PRACFACTOR
        }
        for col in missing_cols:
            st.warning(f"❌ Column '{col_names[col]}' not detected - some analyses may not be available")

st.markdown("---")
st.caption("💡 **Tip**: Use the filters above to focus on specific analysis categories and queries")
st.caption("📊 **CodeGenAI** - Powered by Streamlit | © 2025 Bina Nusantara University")
