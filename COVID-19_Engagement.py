import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# --------------------------
# Load data
# --------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("data/covid_articles_matched_qids.csv")
df["date"] = pd.to_datetime(df["date"])

st.title("Shifts in COVID-19 Global Public Interest")
st.write("**How has public engagement in COVID-19 shifted during the post-pandemic period, as reflected in Wikipedia pageviews from 2023–2024, and what categories of COVID-19-related articles are most popular?**")

st.write("This dashboard explores how interest in COVID-19 evolved since the height of the pandemic. By using Wikipedia pageviews as a proxy of engagement for COVID-19-related" \
" articles from 2023 to 2024, we can measure popularity and interest within this topic over time. This analysis can help us understand whether public attention to COVID-19 has , " \
"retained, or shifted to specific aspects of the pandemic in the post-pandemic period.")

st.write("Through this analysis, I expect to find that overall public interest in COVID-19 has declined over time in the post-pandemic. However, certain topics within the broader COVID-19 context (e.g., vaccine, misinformation, government) still have maintained interest.")

# --------------------------
# Data Summary
# --------------------------
st.subheader("Data Summary")
st.write("The dataset consists of articles from the WikiProject COVID-19 Wikipedia page with their respective pageviews from 02-06-2023 to 12-31-2024. Only articles with top, high, and medium importance levels were included for relevancy. QIDs were matched with all Wikipedia data to get global pageviews.")
st.write("Preview of the raw data:")
st.dataframe(df.head())
st.write(f"Total Articles: {df['article'].count():,}")
st.write(f"Total Pageviews (All Years): {df['pageviews'].sum():,}")
st.write(f"Average Pageviews: {df['pageviews'].mean():.2f}")

# --------------------------
# Total Pageviews Over Time
# --------------------------
st.subheader("Total COVID-19 Pageviews Over Time")

# Date range slider
min_date = df["date"].min().to_pydatetime()
max_date = df["date"].max().to_pydatetime()

start_date, end_date = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

df_filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

# Total daily pageviews
daily = df_filtered.groupby("date")["pageviews"].sum().reset_index()

line_chart = alt.Chart(daily).mark_line().encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y")),
    y=alt.Y("pageviews:Q", title="Total Pageviews"),
    tooltip=["date:T", "pageviews:Q"]
).properties(height=350)

st.altair_chart(line_chart, use_container_width=True)

# --------------------------
# Year Comparison
# --------------------------
st.subheader("2023 vs. 2024 — Total Pageviews")

df["year"] = df["date"].dt.year

yearly = df.groupby("year")["pageviews"].sum().reset_index()

bar_year = alt.Chart(yearly).mark_bar().encode(
    x=alt.X("year:N", title="Year"),
    y=alt.Y("pageviews:Q", title="Total Pageviews"),
    color=alt.Color("year:N", legend=None),
)

st.altair_chart(bar_year, use_container_width=True)

# -------------------------------------------------------
# Monthly Pageviews for Top 10 Articles 
# -------------------------------------------------------
st.subheader("Monthly Pageviews for Top 10 COVID-19 Articles by Year")

# select year from selectbox
year_selected = st.selectbox(
    "Select Year",
    options=[2023, 2024],
    index=0
)

st.write('The top COVID-19 article in 2023, "Coronavirus", has an extreme number of pageviews in February 2023 ' \
'that skews the data visualization. It can be excluded to better visualize trends for other articles.')
exclude_coronavirus = st.checkbox(
    'Exclude "Coronavirus" article', value=False
)

# Filter selected year
df_year = df[df["date"].dt.year == year_selected]
if exclude_coronavirus:
    df_year = df_year[df_year["article"] != "Coronavirus"]

# find top 10 articles for that year
top10_articles = (
    df_year.groupby("article")["pageviews"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

df_top10 = df_year[df_year["article"].isin(top10_articles)]

df_top10["month"] = df_top10["date"].dt.to_period("M").dt.to_timestamp()

# sum monthly pageviews
monthly = (
    df_top10.groupby(["month", "article"])["pageviews"]
    .sum()
    .reset_index()
)

monthly_chart = (
    alt.Chart(monthly)
    .mark_line()
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("pageviews:Q", title="Total Pageviews"),
        color="article:N",
        tooltip=[
            alt.Tooltip("month:T", title="Month"),
            alt.Tooltip("article:N", title="Article"),
            alt.Tooltip("pageviews:Q", title="Pageviews")
        ]
    )
    .properties(height=400)
)

st.altair_chart(monthly_chart, use_container_width=True)

# --------------------------
# 3. Top Articles by Total Views
# --------------------------
st.subheader("Top 10 Most Popular COVID-19 Articles (2023–2024)")

top_articles = (
    df.groupby("article")["pageviews"]
    .sum()
    .reset_index()
    .sort_values("pageviews", ascending=False)
    .head(10)
)

bar = alt.Chart(top_articles).mark_bar().encode(
    x=alt.X("pageviews:Q", title="Total Pageviews"),
    y=alt.Y("article:N", sort='-x', title="Article", axis=alt.Axis(labelLimit=300)),
).properties(height=500)

st.altair_chart(bar, use_container_width=True)

# --------------------------
# Category Analysis by Text Classification
# --------------------------
st.subheader("COVID-19 Articles Category Distribution")

st.write("Articles were classified into one of 12 candidate categories, shown in the table below, using zero-shot text classification. The excerpt chart below displays predicted categories for each article, the probability scores, and the true category (ground truth).")

candidate_categories = [
    "misinformation",
    "vaccine",
    "treatment",
    "lockdown",
    "human",
    "government",
    "facility",
    "response",
    "variant",
    "societal impact",
    "timeline",
    "disease"
]
df_cc = pd.DataFrame({"Categories": candidate_categories})

df_category = load_data("data/predicted_categories.csv")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("#### Categories")
    st.dataframe(df_cc, column_config={
        "Categories": st.column_config.Column(
            width=105 
        )},
        use_container_width=False)

with col2:
    st.markdown("#### Classification Prediction Data") 
    st.dataframe(df_category.head(30), column_config={"article": st.column_config.Column(width=320)}, use_container_width=False)

pred_counts = df_category["predicted_label"].value_counts().reset_index()
pred_counts.columns = ["category", "count"]
pred_counts["type"] = "Predicted"

gt_counts = df_category["ground_truth"].value_counts().reset_index()
gt_counts.columns = ["category", "count"]
gt_counts["type"] = "True"

st.subheader("Predicted Category Counts")
st.dataframe(pred_counts)

st.subheader("Ground Truth Category Counts")
st.dataframe(gt_counts)


combined = pd.concat([pred_counts, gt_counts], ignore_index=True)

# Plotly for grouped bar chart
fig = px.bar(
    combined,
    x="category",
    y="count",
    color="type",
    barmode="group",
    title="Predicted vs True Category Distribution",
    hover_data=["count"],
)

fig.update_layout(
    xaxis_title="Category",
    yaxis_title="Frequency",
    legend_title="Label Type",
    bargap=0.15,
    bargroupgap=0.05
)

st.plotly_chart(fig)

