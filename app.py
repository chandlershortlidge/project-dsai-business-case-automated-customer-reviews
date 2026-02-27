import json
import os

import ollama
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Electronics",
    "Tablets & E-readers",
    "Health & Beauty",
    "Home & Kitchen",
    "Office Supplies",
    "Pet Supplies",
]

SUMMARIES_PATH = os.path.join(os.path.dirname(__file__), "notebooks", "summaries.json")
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "notebooks", "amazon_sentiment_categories.csv"
)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#f39c12",
    "negative": "#e74c3c",
}


def category_key(category: str) -> str:
    """Normalise a category name to a dict key, e.g. 'Health & Beauty' -> 'health_beauty'."""
    return category.lower().replace(" & ", "_").replace("&", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading review data...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_summaries() -> dict:
    if os.path.exists(SUMMARIES_PATH):
        with open(SUMMARIES_PATH) as f:
            return json.load(f)
    return {}


def save_summaries(summaries: dict) -> None:
    with open(SUMMARIES_PATH, "w") as f:
        json.dump(summaries, f, indent=2)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def get_top_n(df: pd.DataFrame, category: str, n: int = 3) -> pd.DataFrame:
    cat_df = df[df["meta_category"] == category]
    return (
        cat_df.groupby("name")
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values("num_reviews", ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"name": "Product", "avg_rating": "Avg Rating", "num_reviews": "# Reviews"})
    )


def get_worst(df: pd.DataFrame, category: str) -> pd.DataFrame:
    cat_df = df[df["meta_category"] == category]
    return (
        cat_df.groupby("name")
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values("avg_rating", ascending=True)
        .head(1)
        .reset_index()
        .rename(columns={"name": "Product", "avg_rating": "Avg Rating", "num_reviews": "# Reviews"})
    )


def get_rating_chart(df: pd.DataFrame, category: str):
    cat_df = df[df["meta_category"] == category]
    product_ratings = (
        cat_df.groupby("name")
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values("num_reviews", ascending=False)
        .head(10)
        .reset_index()
    )
    # Truncate long product names for display
    product_ratings["short_name"] = product_ratings["name"].str[:40] + "..."
    fig = px.bar(
        product_ratings,
        x="avg_rating",
        y="short_name",
        orientation="h",
        title="Avg Rating — Top 10 Products by Review Count",
        labels={"avg_rating": "Avg Rating", "short_name": "Product"},
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        range_color=[1, 5],
        text="avg_rating",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )
    return fig


def get_sentiment_chart(df: pd.DataFrame, category: str):
    cat_df = df[df["meta_category"] == category]
    counts = cat_df["rating_sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.4,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    return fig


# ---------------------------------------------------------------------------
# Blog generation
# ---------------------------------------------------------------------------
def generate_blog(df: pd.DataFrame, category: str) -> str:
    cat_df = df[df["meta_category"] == category]

    top_n = (
        cat_df.groupby(["meta_category", "name"])
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values("num_reviews", ascending=False)
        .head(3)
    )
    top_names = top_n.index.get_level_values("name").tolist()
    top_reviews = [df[df["name"] == name]["reviews.text"].tolist() for name in top_names]

    worst_product = (
        cat_df.groupby(["meta_category", "name"])
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values("avg_rating", ascending=True)
        .head(1)
    )
    worst_name = worst_product.index.get_level_values("name")[0]
    worst_reviews = df[df["name"] == worst_name]["reviews.text"].tolist()

    top_reviews_str = "\n\n".join(
        [f"Product {i + 1} - {top_names[i]}:\n{top_reviews[i]}" for i in range(len(top_names))]
    )

    response = ollama.chat(
        model="qwen2.5",
        messages=[
            {
                "role": "user",
                "content": f"""
Write a short article (like a blog post) about the product category: {category}. The output should include:

- Top 3 products {top_n} and key differences between them.
- Reviews and top complaints for each: {top_reviews_str}

-----
You should also include the worst product {worst_product} in the category and why it should be avoided.
Worst product reviews: {worst_reviews}
Do not forget to include this.
----

Make sure the style is like a blog.
""",
            }
        ],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Review Dashboard",
    page_icon="",
    layout="wide",
)

# Load data once
df = load_data()

# Sidebar
st.sidebar.title("Amazon Review Dashboard")
st.sidebar.markdown("Explore product categories and generate AI-powered blog summaries.")
selected_category = st.sidebar.selectbox("Select Category", CATEGORIES)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**{len(df[df['meta_category'] == selected_category]):,}** reviews in this category"
)

# Main header
st.title(f"{selected_category}")
st.markdown("---")

# ── Charts ──────────────────────────────────────────────────────────────────
chart_col, pie_col = st.columns(2)
with chart_col:
    st.plotly_chart(get_rating_chart(df, selected_category), use_container_width=True)
with pie_col:
    st.plotly_chart(get_sentiment_chart(df, selected_category), use_container_width=True)

# ── Product tables ───────────────────────────────────────────────────────────
st.markdown("---")
top_col, worst_col = st.columns([2, 1])

with top_col:
    st.subheader("Top 3 Products by Review Count")
    top3 = get_top_n(df, selected_category, n=3)
    st.dataframe(
        top3.style.format({"Avg Rating": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

with worst_col:
    st.subheader("Worst Rated Product")
    worst = get_worst(df, selected_category)
    st.dataframe(
        worst.style.format({"Avg Rating": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

# ── Blog summary ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Blog Summary")

key = category_key(selected_category)
session_key = f"summary_{key}"

# Seed session state from saved summaries on first load
if session_key not in st.session_state:
    saved = load_summaries()
    st.session_state[session_key] = saved.get(key, "")

if st.button("Generate Blog Summary with Qwen 2.5", type="primary"):
    with st.spinner("Generating with Qwen 2.5... this may take a minute"):
        try:
            summary = generate_blog(df, selected_category)
            st.session_state[session_key] = summary
            # Persist to summaries.json
            saved = load_summaries()
            saved[key] = summary
            save_summaries(saved)
            st.success("Done! Summary saved to summaries.json")
        except Exception as e:
            st.error(f"Ollama error: {e}")

if st.session_state[session_key]:
    st.markdown(st.session_state[session_key])
else:
    st.info("No summary yet — click the button above to generate one.")
