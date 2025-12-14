import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="COVID-19 Public Interest Analysis",
    layout="wide"
)
st.sidebar.title("Table of Contents")

st.sidebar.markdown("""
- [Overview](#overview)
- [Data Summary](#data-summary)
- [COVID-19 Pageviews Over Time](#total-pageviews)
- [Top 10 COVID-19 Articles](#top-articles)
- [Article Category Classification](#category-classification)
- [Category Popularity Over Time](#category-popularity)
- [Summary](#summary)
""", unsafe_allow_html=True)

@st.cache_data

def load_data():
    df = pd.read_csv("data/covid_articles_matched_qids.csv")
    return df

df = load_data()
df["date"] = pd.to_datetime(df["date"])

st.markdown("<a name='overview'></a>", unsafe_allow_html=True)
st.title("Shifts in COVID-19 Global Public Interest")
st.markdown("#### **How has public engagement in COVID-19 shifted during the post-pandemic period, as reflected in Wikipedia pageviews from 2023–2024?**")

st.write("This dashboard explores how interest in COVID-19 evolved since the height of the pandemic. As of May 2023, the World Health Organization declared an end to the COVID-19 pandemic. By using Wikipedia pageviews as a proxy of engagement for COVID-19-related" \
" articles from 2023 to 2024, we can measure popularity and interest within this topic over time. This analysis can help us understand whether public attention to COVID-19 has declined, " \
"retained, or shifted to specific aspects of COVID-19 in the post-pandemic period.")

st.write('Through this investigation, I expect to find that overall public engagement in COVID-19 has declined post-pandemic from 2023 to 2024. \
However, certain topics within the broader COVID-19 context are also expected to have shifted from categories such as "disease" and "lockdown" to "societal impact" over time as the pandemic period passes.')

st.markdown("""
**See main findings at:**
- [Total Pageviews Over Time](#total-pageviews)
- [Category Popularity Over Time](#category-popularity)
""", unsafe_allow_html=True)


# -------- Data Summary --------
st.markdown("<a name='data-summary'></a>", unsafe_allow_html=True)

st.subheader("Data Summary")
st.write("The dataset consists of articles from the WikiProject COVID-19 Wikipedia page with their respective pageviews from 02-06-2023 to 12-31-2024. Only articles with **top**, **high**, and **medium** importance levels were included for relevancy. QIDs were matched with all Wikipedia data to get global pageviews.")
st.write("Preview of the raw data:")
st.dataframe(df.head())

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown("##### Total Articles:")
    st.write(f"{df['article'].count():,}")

with col2:
    st.markdown("##### Total Pageviews (All Years):")
    st.write(f"{df['pageviews'].sum():,}")

with col3:
    st.markdown("##### Average Pageviews:")
    st.write(f"{df['pageviews'].mean():.2f}")

# -------- Total Pageviews Over Time --------
st.markdown("<a name='total-pageviews'></a>", unsafe_allow_html=True)

st.subheader("Total COVID-19 Pageviews Over Time")

def load_peak_data():
    df = pd.read_csv("data/known_peaks.csv")
    return df

df_peaks = load_peak_data()
df_peaks['date'] = pd.to_datetime(df_peaks['date']) 

st.write("This time series displays total Wikipedia pageviews of COVID-19-related articles from 2023 to 2024. " \
"Prominent peaks are dynamically annotated on the graph with a red circle, and if hovered over, the date of the peak along " \
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

        # Get article contributions on that date 
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
            annotation_text += f"<br>{a['article']}: {percent:.2f}%"
        
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

# Year Comparison 
st.markdown("#### Total Pageviews in 2023 vs. 2024")

st.write('Overall, there are more Wikipedia pageviews and thus higher public interest in COVID-19 in 2023 compared to 2024, mainly due to the extremely high views on the "Coronavirus" article in February 2023. This indicates a decline in public engagement in COVID-19 post-pandemic.')

df["year"] = df["date"].dt.year

yearly = df.groupby("year")["pageviews"].sum().reset_index()

bar_year = alt.Chart(yearly).mark_bar().encode(
    x=alt.X("year:N", title="Year"),
    y=alt.Y("pageviews:Q", title="Total Pageviews"),
    color=alt.Color("year:N", legend=None),
).properties(height=400)

st.altair_chart(bar_year, use_container_width=True)

# -------- Top Articles by Total Views --------
st.markdown("<a name='top-articles'></a>", unsafe_allow_html=True)

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
    color=alt.Color("article:N", title=" ", legend=None)  
).properties(height=500)

st.altair_chart(bar, use_container_width=True)

# Monthly Pageviews for Top 10 Articles 

st.markdown("#### Monthly Pageviews for Top 10 COVID-19 Articles by Year")

st.write('The top COVID-19 article in 2023, "Coronavirus", has an extreme number of pageviews in February 2023 ' \
'that skews the data visualization. It can be excluded to better visualize trends for other articles.')
exclude_coronavirus = st.checkbox(
    'Exclude "Coronavirus" article', value=False
)

# select year from selectbox
year_selected = st.selectbox(
    "Select Year",
    options=[2023, 2024],
    index=0
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
st.markdown("<a name='category-classification'></a>", unsafe_allow_html=True)

st.subheader("COVID-19 Articles Category Classification")

st.write("API calls to the Wikidata database were used to retrieve the label, description, and relevant attributes for each COVID-19 article given its QID. These semantic features were then compiled to assign the category classifcation of the article.")
st.write("To test a text classification technique, articles were classified into one of 12 predicted candidate categories, shown below in the left table, using **zero-shot text classification**. The dataframe on the right displays a preview of the predicted categories for each article, the probability scores, and the true category (ground truth).")
st.write('**Zero-shot classification model**: "facebook/bart-large-mnli" from Hugging Face')
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

st.write('The "other" category resulted from articles that did not fit into any of the candidate categories and were thus placed into this category when classifying ground truth labels. The zero-shot text classifer did not have "other" in the candidate labels, so it seems that those unclear articles were misclassified as mostly "variant"-type articles.')

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

# Text Classification Evaluation

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

st.write('The zero-shot text classification model achieved an accuracy of 67.94%, which is adequate, but definitely can be improved. It seems to have achieved perfect accuracy for classifying articles into the "misinformation", "timeline", and "variant" categories, but did the poorest on classifying "societal impact" articles (excluding "other").')

# -------- Category Popularity of COVID-19 Articles Over Time --------
st.markdown("<a name='category-popularity'></a>", unsafe_allow_html=True)

st.subheader('Category Popularity of COVID-19 Articles Over Time')

st.write('The below visualizations depict how popularity and interest in specific COVID-19 article categories have changed from 2023 to 2024. The most notable trend is the sharp decrease in the "disease" category from February to March 2023, largely due to the exceedingly high pageviews from the "Coronavirus" article (classified as "disease") on February 24, 2023.' \
' The "disease" articles stay relatively high and consistent in pageviews for the rest of the time period, and no other categories seemed to have significantly different patterns over time. However, "human" articles appear to have increased pageviews during October 2023. Additionally, pageviews for articles in the "societal impact" category are also relatively consistent from 2023 to 2024.')
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
    color='ground_truth',
    labels={'ground_truth': 'Category'}
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
    animation_frame=category_monthly['month'].dt.strftime('%b %Y'),
    range_y=[0, category_monthly['pageviews'].max()]
)

fig.update_layout(
    xaxis_title='Category',
    yaxis_title='Total Pageviews',
    sliders=[{
        "y": -0.05, # move closer to plot
        "x": 0.1,
        "len": 0.85
    }]
)

st.plotly_chart(fig, use_container_width=True)

# --------  Summary and Ethical Considerations --------
st.markdown("<a name='summary'></a>", unsafe_allow_html=True)

st.subheader("Summary")

st.markdown("#### Key Takeaways")
st.write('This analysis investigated how public interest in COVID-19 topics has changed during the post-pandemic era, specifically from 2023 to 2024, ' \
'using Wikipedia pageview data of COVID-19 related articles. Additional features were examined as well, such as category classification prediction with zero-shot text classification. ' \
'Ultimately, the analysis reveals that public engagement during the post-pandemic period has generally decreased, and that interest remains unevenly distributed across topics. This partially supports my hypothesis' \
' that overall attention in COVID-19 will drop over time, but interest towards specific aspects of COVID-19 will have shifted from more medical crisis-like topics to articles focusing on current impacts. ' \
'It is unclear if this shift occured as an ambiguous distrubution of interest in topic categories is observed over time.')

st.markdown("#### Limitations")
st.write("Although this analysis is true to numbers as it uses direct real-time data from Wikipedia, utilizing pageviews as a proxy for interest poses many limitations. For example, Wikipedia was chosen as it is one of the world's most accessible sites for information, yet it is still not available in some countries or accessible to those who do not have the abilties to search the web. " \
" Additionally, traffic accrued may not represent actual interest and engagement in the article.")
st.write("Moreover, the COVID-19 articles collected from WikiProject COVID-19 for this analysis only included top, high, and medium importance articles for the sake of relevancy and efficiency. The majority of articles in the list are of low or unrated importance. This means the dataset is incomplete and may bias results toward more prominent topics.")
st.write("Classifying ground truth labels for the COVID-19 articles was difficult as the Wikidata results had different instances for every article. Therefore, ground truth categories were dervied from keyword extraction and some manual interpretation. Inconsistencies in these labels will affect analyses that use this data.")

st.markdown("#### Reliability and Accuracy")
st.write('Zero-shot text classification was used for predicting COVID-19 article categories. Results indicated that the classifier model did well for more clearly-defined topics like "variant", but less so for broader and overlapping topics like "societal impact" and "response". While this classification is sufficient, it should not be used as a completely reliable model.')

st.markdown("#### Ethical Considerations")
st.write("As stated, Wikipedia is not accessible for everyone globally, so Wikipedia use reflects biases in internet access, demographic, and geography/region. Most Wikipedia users are English speakers from the United States, disproportionately representing Wikipedia usage.")
st.write("Zero-shot classification models may be biased depending on the data they are trained on, which may produce skewed or misleading outputs. Another is bias towards seen or known classes; the model will missclassify unseen classes into a seen class, which is what occurred in this analysis.")
