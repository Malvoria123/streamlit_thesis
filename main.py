# import modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations

st.set_page_config(layout="wide", page_title="AI & Programming Insights Explorer", page_icon="🤖")

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
            quotechar='"',           # Character used to quote fields
            escapechar='\\',         # Escape character
            encoding='utf-8',        # Try utf-8 first
            on_bad_lines='skip'      # Skip truly malformed lines
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

# Streamlit UI

st.title("🤖 Generative AI & Programming Insights Explorer")
st.markdown("An Interactive Application for Analysis on Generative AI Based Programming Tools")
st.markdown("🔗 Data Source: Reddit")

# sidebar configuration
st.sidebar.title("⚙️ Configuration")

# owner information
with st.sidebar.expander("👥 About Us"):
    st.markdown("### 📚 Non-Class Thesis")
    st.markdown("**Bina Nusantara University**")
    st.markdown("---")
    for owner in OWNERS:
        st.markdown(f"**{owner['name']}**")
        st.caption(f"NIM: {owner['nim']}")
        st.markdown("")

# dataset statistics
with st.sidebar.expander("📊 Dataset Statistics"):
    st.metric("Number of Rows", f"{len(df):,}")
    st.metric("Total Columns", len(available_cols))
    if col_presence['ai']:
        total_ai_mentions = sum(len(x) for x in list_ai)
        st.metric("Total AI Mentions", f"{total_ai_mentions:,}")

# data preview
with st.expander("👀 Dataset Preview (10 First Rows)"):
    if col_presence['text']:
        preview_df = df[[COL_TEXT]].head(10).copy()
        preview_df.index = range(1, len(preview_df) + 1)
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.warning(f"Column '{COL_TEXT}' not found")
        st.write("Available columns:", available_cols)

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

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter Analysis")
selected_sections = st.sidebar.multiselect(
    "Choose Analysis Category",
    options=list(section_options.keys()),
    default=list(section_options.keys())
)

# build query selection
available_queries = []
for sec in selected_sections:
    available_queries.extend(section_options[sec])

if not available_queries:
    st.warning("⚠️ Choose at least one category in sidebar to continue")
    st.stop()

query_label_map = {q[0]: q[1] for q in available_queries}
query_keys = [q[0] for q in available_queries]

st.sidebar.markdown("---")
query_choice = st.sidebar.selectbox(
    "Choose Analysis Query",
    options=[(k + ' - ' + query_label_map[k]) for k in query_keys],
    help="Choose the analysis question to be answered"
)

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
    ai_choice = st.selectbox(
        "🤖 Choose AI Tool",
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
    lang_choice = st.selectbox(
        "💻 Choose Programming Language",
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
    ai_choice = st.selectbox(
        "🤖 Choose AI Tool",
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

# perform analysis
selected_q_key = query_choice.split(' - ')[0]


with st.spinner("🔄 Processing analysis..."): # auto-execute
    func = answer_map.get(selected_q_key)
    if func is None:
        st.error("Query not yet implemented.")
    else:
        try:
            chart, table, explanation = func(df)
            
            
            st.markdown("---")
            st.subheader(f"📊 Analysis Results: {query_label_map[selected_q_key]}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("📉 Chart not available: " + (explanation or "No data available."))
            
            with col2:
                st.markdown("### 💡 Insight")
                st.info(explanation)
                
                if table is not None and not table.empty:
                    st.markdown("### 📈 Quick Stats")
                    # Total Items = number of unique items in ALL data
                    st.metric("Total Items", f"{len(table):,}", 
                             help="Number of unique entities found in dataset")
                    
                    # Total Mentions = sum of 'Count' column for ALL data
                    if 'Count' in table.columns:
                        total_mentions = table['Count'].sum()
                        st.metric("Total Mentions", f"{total_mentions:,}",
                                 help="Total occurrences of all items (including duplicates)")
            
            # detailed table - SHOWING ALL DATA
            if table is not None and not table.empty:
                with st.expander(f"📋 View Detailed Data (Total: {len(table):,} items)"):
                    st.dataframe(table, use_container_width=True, height=400)
                    
                    # download option
                    csv = table.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Complete CSV",
                        data=csv,
                        file_name=f"analysis_{selected_q_key}_complete.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Table not available: " + (explanation or "No data available."))
        
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            with st.expander("🐛 Error Details"):
                st.exception(e)

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
st.caption("💡 **Tip**: Use filters in sidebar to focus on specific analysis categories")