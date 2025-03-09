import streamlit as st
import pandas as pd
import ast
import itertools
import json
import umap
import plotly.express as px
import numpy as np
import hdbscan
import openai
import os
import markdown
from dotenv import load_dotenv
import re

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def intersect_lists(lists):
    if not lists:
        return []
    return list(set.intersection(*[set(item) for item in lists]))

df = pd.read_csv("clustered_df_streamlit.csv")
df_summaries = pd.read_parquet("summarized_summaries.parquet")

for col in ['key_themes', 'recurring_topics']:
    df[col] = df[col].apply(ast.literal_eval)

with open("collection_export.json", "r") as f:
    json_data = json.load(f)

embeddings_df = pd.DataFrame({
    "embedding": json_data["embeddings"],
    "source": [meta["source"] for meta in json_data["metadatas"]],
    "title": [meta.get("title", "") for meta in json_data["metadatas"]],
    "date": [meta.get("date", "") for meta in json_data["metadatas"]],
    "category": [meta.get("category", "") for meta in json_data["metadatas"]],
    "niche": [meta.get("niche", "") for meta in json_data["metadatas"]],
    "key_themes": [meta.get("key_themes", "") for meta in json_data["metadatas"]],
    "recurring_topics": [meta.get("recurring_topics", "") for meta in json_data["metadatas"]],
    "document": json_data["documents"],
    "id": json_data["ids"]
})

reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings_df["embedding"].tolist())

embeddings_df["x"] = embeddings_2d[:, 0]
embeddings_df["y"] = embeddings_2d[:, 1]

X = np.array(embeddings_df["embedding"].tolist())
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(X)
embeddings_df["cluster"] = cluster_labels

# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------
categories = df['category'].unique()
selected_category = st.sidebar.selectbox("Select a Category", categories)

filter_by_cluster = st.sidebar.checkbox("Filter by Cluster?")

unique_niches = df['niche'].unique().tolist()
niche_options = ["None"] + sorted(unique_niches)
selected_niche = st.sidebar.selectbox("Select a Niche (Optional)", niche_options)

min_shared_n = st.sidebar.slider(
    "Minimum number of shared themes or topics",
    min_value=1,
    max_value=5,
    value=1,
    help="A subset is shown if it has at least this many shared themes OR this many shared topics."
)

max_subset_size = st.sidebar.slider(
    "Maximum subset size (to avoid huge combinations)",
    min_value=2,
    max_value=10,
    value=5,
    help="Larger subsets might produce too many combinations if you have many articles."
)

# ------------------------------------------------------
# FILTER THE DATA
# ------------------------------------------------------
filtered_df = df[df['category'] == selected_category]
if selected_niche != "None":
    filtered_df = filtered_df[filtered_df['niche'] == selected_niche]

# ------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------
def clean_markdown_fences(md_text: str) -> str:
    """
    Removes ```markdown or ``` from the start/end of the string if present.
    """
    cleaned = re.sub(r"```(?:markdown)?", "", md_text)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned

def get_subsets_of_articles(articles, min_size=2, max_size=5):
    n = len(articles)
    for size in range(min_size, min(max_size, n) + 1):
        for combo in itertools.combinations(range(n), size):
            yield [articles[i] for i in combo]

def get_shared_items(subset, col_name):
    if not subset:
        return set()
    sets = [set(article[col_name]) for article in subset]
    return set.intersection(*sets)

def display_subset(subset, shared_themes, shared_topics):
    st.markdown("**Subset of {} articles**:".format(len(subset)))
    for art in subset:
        st.markdown(f"- **{art['title']}** by **{art['source']}**")
    st.markdown("")

    if shared_themes:
        st.markdown(f"**Shared Themes ({len(shared_themes)})**: {', '.join(sorted(shared_themes))}")
    else:
        st.markdown("**Shared Themes:** None")

    if shared_topics:
        st.markdown(f"**Shared Topics ({len(shared_topics)})**: {', '.join(sorted(shared_topics))}")
    else:
        st.markdown("**Shared Topics:** None")

    st.write("---")

# ------------------------------------------------------
# PLACEHOLDER FOR COMPARATIVE ANALYSIS FUNCTION
# ------------------------------------------------------
def get_comparative_analysis(summary1, summary2):
    prompt = (
        "Please provide a detailed comparative analysis of the two summaries provided below. "
        "Compare and contrast the underlying points, thoughts, and insights. "
        "Return the result in Markdown format that is well-formatted and visually appealing for Streamlit.\n\n"
        f"**Summary 1:**\n{summary1}\n\n"
        f"**Summary 2:**\n{summary2}\n\n"
        "Comparative Analysis:"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    analysis = response.choices[0].message.content.strip()
    return analysis

# ------------------------------------------------------
# SESSION STATE SETUP & QUERY PARAMS
# ------------------------------------------------------
if "selected_articles" not in st.session_state:
    st.session_state["selected_articles"] = []

query_params = st.query_params
default_tab = query_params.get("tab", ["similarity"])[0]

# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
st.title("Benori Assignment - Amal Singh")

# ------------------------------------------------------
# TABS SETUP
# ------------------------------------------------------.
if default_tab == "comparative":
    tabs = st.tabs(["📊 Comparative Analysis", "🔍 Similarity Analysis", "🌐 Cluster Visualization"])
elif default_tab == "visualization":
    tabs = st.tabs(["🌐 Cluster Visualization", "🔍 Similarity Analysis", "📊 Comparative Analysis"])
else:
    tabs = st.tabs(["🔍 Similarity Analysis", "📊 Comparative Analysis", "🌐 Cluster Visualization"])

# ------------------------------------------------------
# TAB 1: Similarity Analysis
# ------------------------------------------------------
# --- Similarity Analysis Tab ---
with tabs[0] if default_tab == "similarity" else tabs[1]:
    st.title(f"Descriptive Analytics for Category: {selected_category}")
    if selected_niche != "None":
        st.write(f"**Selected Niche:** {selected_niche}")
    else:
        st.write("**No specific niche selected** (using only Category filter).")

    st.header("1. Articles Count by Source")
    if filtered_df.empty:
        st.warning("No articles found for this combination of Category and Niche.")
        st.stop()
    else:
        source_counts = filtered_df['source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Article Count']
        st.dataframe(source_counts.reset_index(drop=True))
        st.bar_chart(source_counts.set_index('Source'))

    st.header("2. Subsets of Articles With Shared Themes/Topics")

    if filter_by_cluster:
        for cluster_val, group in filtered_df.groupby('cluster'):
            st.subheader(f"Cluster: {cluster_val}")
            articles_in_cluster = group.to_dict('records')
            st.markdown("**Articles in this cluster:**")
            for row in articles_in_cluster:
                st.markdown(f"- **{row['title']}** by **{row['source']}**")

            if len(articles_in_cluster) < 2:
                st.warning("Fewer than 2 articles in this cluster, so no subsets.")
                st.write("---")
                continue

            found_any = False
            for subset in get_subsets_of_articles(articles_in_cluster, 2, max_subset_size):
                shared_themes = get_shared_items(subset, 'key_themes')
                shared_topics = get_shared_items(subset, 'recurring_topics')
                if len(shared_themes) >= min_shared_n:
                    found_any = True
                    display_subset(subset, shared_themes, shared_topics)

                    # subset_key = "_".join([art['title'][:10] for art in subset])
                    # if st.button(f"Compare and Analyze ({len(subset)} Articles)", key=f"compare_{subset_key}"):
                    #     st.session_state["selected_articles"] = subset
                    #     st.set_query_params(tab="comparative")
                    #     st.experimental_rerun()

            if not found_any:
                st.info("No subsets of articles share enough themes/topics in this cluster.")
            st.write("---")
    else:
        st.subheader("All Articles (Ignoring Clusters)")
        articles_filtered = filtered_df.to_dict('records')
        st.markdown("**Articles in this selection:**")
        for row in articles_filtered:
            st.markdown(f"- **{row['title']}** by **{row['source']}**")

        if len(articles_filtered) < 2:
            st.warning("Fewer than 2 articles in this selection, so no subsets.")
            st.stop()

        found_any = False
        for subset in get_subsets_of_articles(articles_filtered, 2, max_subset_size):
            shared_themes = get_shared_items(subset, 'key_themes')
            shared_topics = get_shared_items(subset, 'recurring_topics')
            if len(shared_themes) >= min_shared_n:
                found_any = True
                display_subset(subset, shared_themes, shared_topics)

                # subset_key = "_".join([art['title'][:10] for art in subset])
                # if st.button(f"Compare and Analyze ({len(subset)} Articles)", key=f"compare_{subset_key}"):
                #     st.session_state["selected_articles"] = subset
                #     st.set_query_params(tab="comparative")
                #     st.experimental_rerun()

        if not found_any:
            st.info("No subsets of articles share enough themes/topics in this selection.")

# ------------------------------------------------------
# TAB 2: Comparative Analysis (Updated)
# ------------------------------------------------------
with tabs[1] if default_tab == "similarity" else tabs[0]:
    st.title("📊 Comparative Analysis of Reports")

    report_titles = df_summaries['title'].unique().tolist()
    report1_title = st.selectbox("Select Report 1", report_titles, key="report1")
    report2_title = st.selectbox("Select Report 2", report_titles, key="report2")

    if report1_title == report2_title:
        st.warning("Please select two different reports for comparison.")
    else:
        try:
            summary1 = df_summaries.loc[df_summaries['title'] == report1_title, "condensed_summary"].iloc[0]
            summary2 = df_summaries.loc[df_summaries['title'] == report2_title, "condensed_summary"].iloc[0]
        except IndexError:
            st.error("Unable to find summaries for the selected titles. Please check your data.")
            st.stop()

        if st.button("Generate Comparative Analysis"):
            with st.spinner("Generating comparative analysis..."):
                comparative_result = get_comparative_analysis(summary1, summary2)
            comparative_result = clean_markdown_fences(comparative_result)
            st.markdown(comparative_result)


# ------------------------------------------------------
# TAB 3: Cluster Visualization
# ------------------------------------------------------
with tabs[2]:
    st.title("🌐 Cluster Visualization")

    color_options = st.selectbox(
        "Select color coding based on",
        options=["category", "cluster", "source"],
        index=0
    )

    hover_options = st.multiselect(
        "Select data to show on hover",
        options=["title", "date", "source", "category", "niche"],
        default=["title", "date", "source", "category", "niche"]
    )

    fig = px.scatter(
        embeddings_df, x="x", y="y",
        color=color_options,
        hover_data=hover_options,
        title=f"Embedding Clusters Visualization (by {color_options.capitalize()})",
        color_continuous_scale="Viridis" if color_options == "cluster" else None
    )

    fig.update_layout(
        legend_title=color_options.capitalize(),
        width=854,
        height=480
    )

    st.plotly_chart(fig)
