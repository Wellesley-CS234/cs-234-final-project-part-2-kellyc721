import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import find_peaks

st.set_page_config(
    layout="wide"
)

# --------------------------
# Load data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/covid_articles_matched_qids.csv")
    return df

df = load_data()
df["date"] = pd.to_datetime(df["date"])

st.title("Shifts in COVID-19 Global Public Interest")
st.write("**How has public engagement in COVID-19 shifted during the post-pandemic period, as reflected in Wikipedia pageviews from 2023–2024, and what categories of COVID-19-related articles are most popular?**")

st.write("This dashboard explores how interest in COVID-19 evolved since the height of the pandemic. By using Wikipedia pageviews as a proxy of engagement for COVID-19-related" \
" articles from 2023 to 2024, we can measure popularity and interest within this topic over time. This analysis can help us understand whether public attention to COVID-19 has , " \
"retained, or shifted to specific aspects of the pandemic in the post-pandemic period.")

st.write('Through this analysis, I expect to find that overall public interest in COVID-19 has declined post-pandemic from 2023 to 2024. \
However, certain topics within the broader COVID-19 context are also expected to have shifted from categories such as "disease" and "lockdown" to "societal impact" over time as the pandemic period passes.')

# --------------------------
# Data Summary
# --------------------------
st.subheader("Data Summary")
st.write("The dataset consists of articles from the WikiProject COVID-19 Wikipedia page with their respective pageviews from 02-06-2023 to 12-31-2024. Only articles with top, high, and medium importance levels were included for relevancy. QIDs were matched with all Wikipedia data to get global pageviews.")
st.write("Preview of the raw data:")
st.dataframe(df.head())

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.write("**Total Articles:**")
    st.write(f"{df['article'].count():,}")

with col2:
    st.write("**Total Pageviews (All Years):**")
    st.write(f"{df['pageviews'].sum():,}")

with col3:
    st.write("**Average Pageviews:**")
    st.write(f"{df['pageviews'].mean():.2f}")

# --------------------------
# Total Pageviews Over Time
# --------------------------
st.subheader("Total COVID-19 Pageviews Over Time")

def load_peak_data():
    df = pd.read_csv("data/known_peaks.csv")
    return df

df_peaks = load_peak_data()
df_peaks['date'] = pd.to_datetime(df_peaks['date']) 

st.write("This time series displays total Wikipedia pageviews of COVID-19-related articles from 2023 to 2024. " \
"Prominent peaks are dynamically annotated on the graph with a red circle and if hovered over, the date of the peak along " \
"with the top 3 articles that contributed to this pageview spike will appear.")
annotate_peaks = st.checkbox("Show Prominent Peaks", value=True)

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
df_peaks_filtered = df_peaks[(df_peaks['date'] >= start_date) & (df_peaks['date'] <= end_date)]

# Total daily pageviews
daily = df_filtered.groupby("date")["pageviews"].sum().reset_index()

fig = px.line(daily, x='date', y='pageviews', 
              labels={'date': 'Date', 'pageviews': 'Total Pageviews'})

# Dynamic Peak Annotation w/ Hover (Plotly)
if annotate_peaks and not df_peaks_filtered.empty:
    
    hover_texts = []
    for _, row in df_peaks_filtered.iterrows():
        peak_date = row["date"]
        peak_total_views = row["pageviews"]

        # --- Get article contributions on that date ---
        daily_articles = df[df["date"] == peak_date]

        article_summary = (
            daily_articles
            .groupby("article", as_index=False)["pageviews"]
            .sum()
            .sort_values("pageviews", ascending=False)
        )

        top = article_summary.head(3)
        
        # Build the multi-line text using HTML breaks (<br>)
        annotation_text = (
            f"<b>Peak: {peak_date.strftime('%b %d, %Y')}</b>"
            f"<br>{int(peak_total_views):,} views"
        )

        for _, a in top.iterrows():
            percent = (a["pageviews"] / peak_total_views) * 100 if peak_total_views > 0 else 0
            annotation_text += f"<br>{a['article']}: {percent:.0f}%"
        
        hover_texts.append(annotation_text)

    # Add peak markers with hover info
    fig.add_trace(go.Scatter(
        x=df_peaks_filtered['date'],
        y=df_peaks_filtered['pageviews'],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Peak',
        hoverinfo='text',      # only show hover text
        hovertext=hover_texts  # custom hover text
    ))

    fig.update_traces(
        hoverlabel=dict(
            bgcolor="salmon",
            bordercolor="red",
            font_size=12,
            font_color="black",
        )
    )

    fig.update_layout()

st.plotly_chart(fig, use_container_width=True)

# -------- Year Comparison ---------

st.subheader("Total Pageviews in 2023 vs. 2024")

df["year"] = df["date"].dt.year

yearly = df.groupby("year")["pageviews"].sum().reset_index()

bar_year = alt.Chart(yearly).mark_bar().encode(
    x=alt.X("year:N", title="Year"),
    y=alt.Y("pageviews:Q", title="Total Pageviews"),
    color=alt.Color("year:N", legend=None),
).properties(height = 400)

st.altair_chart(bar_year, use_container_width=True)

# -------- Top Articles by Total Views --------

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
    color=alt.Color("article:N", legend=None)  
).properties(height=500)

st.altair_chart(bar, use_container_width=True)

# -------- Monthly Pageviews for Top 10 Articles --------

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
        color=alt.Color(
            "article:N",
            legend=alt.Legend(
                title="Article",
                labelLimit=200,
                labelFontSize=12,    
                symbolLimit=100    
            )
        ),
        tooltip=[
            alt.Tooltip("month:T", title="Month"),
            alt.Tooltip("article:N", title="Article"),
            alt.Tooltip("pageviews:Q", title="Pageviews")
        ]
    )
    .properties(height=400)
)

st.altair_chart(monthly_chart, use_container_width=True)

# -------- Category Analysis by Text Classification --------

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

def load_category_data():
    df = pd.read_csv("data/predicted_categories.csv")
    return df

df_category = load_category_data()

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("#### Categories")
    st.dataframe(df_cc, column_config={
        "Categories": st.column_config.Column(
            width=150
        )},
        use_container_width=False)

with col2:
    st.markdown("#### Classification Prediction Data") 
    st.dataframe(df_category.head(10), column_config={
        "article": st.column_config.Column(width=500)},
        use_container_width=False)

pred_counts = df_category["predicted_label"].value_counts().reset_index()
pred_counts.columns = ["category", "count"]
pred_counts["type"] = "Predicted"

gt_counts = df_category["ground_truth"].value_counts().reset_index()
gt_counts.columns = ["category", "count"]
gt_counts["type"] = "True"

combined = pd.concat([pred_counts, gt_counts], ignore_index=True)

st.markdown("#### Predicted vs. True Category Distribution of Articles")
# plotly for grouped bar chart
fig = px.bar(
    combined,
    x="category",
    y="count",
    color="type",
    barmode="group",
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

st.markdown("#### Text Classification Evaluation")

df_category["correct"] = (df_category["predicted_label"] == df_category["ground_truth"]).astype(int)

accuracy = (
    df_category.groupby("ground_truth")["correct"].mean().reset_index()
)

accuracy.columns = ["Category", "Accuracy"]

st.dataframe(accuracy.sort_values(by="Accuracy", ascending=False))

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    st.markdown("##### Accuracy:")
    st.write('0.6794')
with col2:
    st.markdown("##### Recall:")
    st.write("0.6454")
with col3:
    st.markdown("##### Precision:")
    st.write("0.6809")
with col4:
    st.markdown("##### F1-Score:")
    st.write('0.5804')

st.markdown('#### Category Popularity of COVID-19 Articles Over Time')

def load_cat_pop_data():
    df = pd.read_csv("data/categories_pageviews.csv")
    return df

cat_pgviews = load_cat_pop_data()
cat_pgviews['date'] = pd.to_datetime(cat_pgviews['date'])

cat_pgviews['month'] = cat_pgviews['date'].dt.to_period('M').dt.to_timestamp()

category_monthly = (
    cat_pgviews.groupby(['month', 'ground_truth'], as_index=False)['pageviews']
      .sum()
)

category_monthly['total'] = (
    category_monthly.groupby('month')['pageviews'].transform('sum')
)

fig = px.area(
    category_monthly,
    x='month',
    y='pageviews',          
    color='ground_truth'
)

fig.update_layout(
    xaxis_title='Month',
    yaxis_title="Total Pageviews",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

fig = px.bar(
    category_monthly,
    x='ground_truth',
    y='pageviews',
    animation_frame=category_monthly['month'].dt.strftime('%Y-%m'),
    range_y=[0, category_monthly['pageviews'].max()]
)

st.plotly_chart(fig, use_container_width=True)



