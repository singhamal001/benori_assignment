import streamlit as st
import pandas as pd
import ast
from functools import reduce
import itertools
from collections import Counter, defaultdict


# Function to compute the intersection of lists
def intersect_lists(lists):
    if not lists:
        return []
    return list(set.intersection(*[set(item) for item in lists]))

df = pd.read_csv("clustered_df_streamlit.csv")

# Convert string representations of lists to actual lists
for col in ['key_themes', 'recurring_topics']:
    df[col] = df[col].apply(ast.literal_eval)

# ------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------
# 1) Category (Mandatory)
categories = df['category'].unique()
selected_category = st.sidebar.selectbox("Select a Category", categories)

# 2) Cluster-based Filtering (Optional)
filter_by_cluster = st.sidebar.checkbox("Filter by Cluster?")

# 3) Niche-based Filtering (Optional)
unique_niches = df['niche'].unique().tolist()
niche_options = ["None"] + sorted(unique_niches)
selected_niche = st.sidebar.selectbox("Select a Niche (Optional)", niche_options)

# 4) Minimum Shared Items (Themes/Topics)
min_shared_n = st.sidebar.slider(
    "Minimum number of shared themes or topics",
    min_value=1,
    max_value=5,
    value=1,
    help="A subset is shown if it has at least this many shared themes OR this many shared topics."
)

# 5) Maximum Subset Size (to prevent exponential blow-up)
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
def get_subsets_of_articles(articles, min_size=2, max_size=5):
    """
    Generate all subsets of `articles` from size `min_size` up to `max_size`.
    Each subset is returned as a list of dict (each dict representing one article row).
    """
    n = len(articles)
    for size in range(min_size, min(max_size, n) + 1):
        for combo in itertools.combinations(range(n), size):
            yield [articles[i] for i in combo]

def get_shared_items(subset, col_name):
    """
    Given a subset (list of article dicts) and a column name ('key_themes' or 'recurring_topics'),
    compute the intersection of that column across all articles in the subset.
    """
    if not subset:
        return set()
    sets = [set(article[col_name]) for article in subset]
    return set.intersection(*sets)

def display_subset(subset, shared_themes, shared_topics):
    """
    Display a subset of articles and their shared themes/topics.
    """
    # Titles & Sources
    st.markdown("**Subset of {} articles**:".format(len(subset)))
    for art in subset:
        st.markdown(f"- **{art['title']}** by **{art['source']}**")
    st.markdown("")
    
    if shared_themes:
        st.markdown(f"**Shared Themes ({len(shared_themes)})**: {', '.join(sorted(shared_themes))}")
    else:
        st.markdown("**Shared Themes**: None")
    
    if shared_topics:
        st.markdown(f"**Shared Topics ({len(shared_topics)})**: {', '.join(sorted(shared_topics))}")
    else:
        st.markdown("**Shared Topics**: None")
    
    st.write("---")

# ------------------------------------------------------
# MAIN DISPLAY
# ------------------------------------------------------
st.title(f"Descriptive Analytics for Category: {selected_category}")
if selected_niche != "None":
    st.write(f"**Selected Niche:** {selected_niche}")
else:
    st.write("**No specific niche selected** (using only Category filter).")

# 1) Articles Count by Source
st.header("1. Articles Count by Source")
if filtered_df.empty:
    st.warning("No articles found for this combination of Category and Niche.")
    st.stop()
else:
    source_counts = filtered_df['source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Article Count']
    st.dataframe(source_counts.reset_index(drop=True))
    st.bar_chart(source_counts.set_index('Source'))

# 2) Finding Subsets That Share At Least `min_shared_n` Themes or Topics
st.header("2. Subsets of Articles With Shared Themes/Topics")

if filter_by_cluster:
    # Group by cluster
    for cluster_val, group in filtered_df.groupby('cluster'):
        st.subheader(f"Cluster: {cluster_val}")
        
        # List articles in this cluster
        articles_in_cluster = group.to_dict('records')
        st.markdown("**Articles in this cluster:**")
        for row in articles_in_cluster:
            st.markdown(f"- **{row['title']}** by **{row['source']}**")
        
        if len(articles_in_cluster) < 2:
            st.warning("Fewer than 2 articles in this cluster, so no subsets.")
            st.write("---")
            continue
        
        # Generate subsets
        found_any = False
        for subset in get_subsets_of_articles(articles_in_cluster, 2, max_subset_size):
            shared_themes = get_shared_items(subset, 'key_themes')
            shared_topics = get_shared_items(subset, 'recurring_topics')
            
            # If intersection of themes or topics is at least min_shared_n
            if len(shared_themes) >= min_shared_n:
                found_any = True
                display_subset(subset, shared_themes, shared_topics)
        
        if not found_any:
            st.info("No subsets of articles share enough themes/topics in this cluster.")
        
        st.write("---")

else:
    # Treat all filtered articles as a single group
    st.subheader("All Articles (Ignoring Clusters)")
    
    articles_filtered = filtered_df.to_dict('records')
    st.markdown("**Articles in this selection:**")
    for row in articles_filtered:
        st.markdown(f"- **{row['title']}** by **{row['source']}**")
    
    if len(articles_filtered) < 2:
        st.warning("Fewer than 2 articles in this selection, so no subsets.")
        st.stop()
    
    # Generate subsets
    found_any = False
    for subset in get_subsets_of_articles(articles_filtered, 2, max_subset_size):
        shared_themes = get_shared_items(subset, 'key_themes')
        shared_topics = get_shared_items(subset, 'recurring_topics')
        
        # If intersection of themes or topics is at least min_shared_n
        if len(shared_themes) >= min_shared_n:
            found_any = True
            display_subset(subset, shared_themes, shared_topics)
    
    if not found_any:
        st.info("No subsets of articles share enough themes/topics in this selection.")