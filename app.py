import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="✈️",
    layout="wide"
)

DATA_PATH = "data/cleaned_tourism_dataset.csv"
REG_MODEL_PATH = "models/best_regression_model.joblib"
CLF_MODEL_PATH = "models/best_classification_model.joblib"


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


df = load_data()
reg_model = load_model(REG_MODEL_PATH)
clf_model = load_model(CLF_MODEL_PATH)

st.title("✈️ Tourism Experience Analytics")
st.caption("Classification, Rating Prediction and Attraction Recommendation")

if df is None:
    st.warning(
        "Dataset not found. Run the Google Colab notebook and place "
        "`cleaned_tourism_dataset.csv` inside the `data/` folder."
    )
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "⭐ Rating Prediction", "🧳 Visit Mode", "🎯 Recommendations"]
)

with tab1:
    st.subheader("Tourism Analytics Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")

    if "UserId" in df.columns:
        c2.metric("Unique Users", f"{df['UserId'].nunique():,}")

    if "AttractionId" in df.columns:
        c3.metric("Unique Attractions", f"{df['AttractionId'].nunique():,}")

    if "Rating" in df.columns:
        st.subheader("Rating Distribution")
        st.bar_chart(df["Rating"].value_counts().sort_index())

    if "VisitMode" in df.columns:
        st.subheader("Visit Mode Distribution")
        st.bar_chart(df["VisitMode"].value_counts())

    attraction_col = "Attraction" if "Attraction" in df.columns else "AttractionId"
    if attraction_col in df.columns:
        st.subheader("Top Attractions")
        top = df[attraction_col].value_counts().head(10)
        st.bar_chart(top)

with tab2:
    st.subheader("Predict Attraction Rating")

    if reg_model is None:
        st.info(
            "Regression model not found. Run the Google Colab notebook and copy "
            "`best_regression_model.joblib` into the `models/` folder."
        )
    else:
        st.write("Enter available feature values to estimate an attraction rating.")

        feature_data = {}

        possible_features = [
            "VisitYear", "VisitMonth", "AttractionId", "UserId",
            "AttractionTypeId", "ContinentId", "RegionId",
            "CountryId", "CityId", "AttractionPopularity"
        ]

        available = [c for c in possible_features if c in df.columns]

        for col in available:
            if pd.api.types.is_numeric_dtype(df[col]):
                default = float(df[col].median())
                feature_data[col] = st.number_input(
                    col,
                    value=default,
                    key=f"reg_{col}"
                )

        if st.button("Predict Rating"):
            try:
                input_df = pd.DataFrame([feature_data])
                prediction = reg_model.predict(input_df)[0]
                prediction = max(1, min(5, prediction))
                st.success(f"Predicted Rating: ⭐ {prediction:.2f} / 5")
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab3:
    st.subheader("Predict Visit Mode")

    if clf_model is None:
        st.info(
            "Classification model not found. Run the notebook and copy "
            "`best_classification_model.joblib` into the `models/` folder."
        )
    else:
        feature_data = {}

        possible_features = [
            "VisitYear", "VisitMonth", "AttractionId", "UserId",
            "AttractionTypeId", "ContinentId", "RegionId",
            "CountryId", "CityId", "AttractionPopularity"
        ]

        available = [c for c in possible_features if c in df.columns]

        for col in available:
            if pd.api.types.is_numeric_dtype(df[col]):
                default = float(df[col].median())
                feature_data[col] = st.number_input(
                    col,
                    value=default,
                    key=f"clf_{col}"
                )

        if st.button("Predict Visit Mode"):
            try:
                input_df = pd.DataFrame([feature_data])
                prediction = clf_model.predict(input_df)[0]
                st.success(f"Predicted Visit Mode: 🧳 {prediction}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab4:
    st.subheader("Personalized Attraction Recommendations")

    if not {"UserId", "AttractionId", "Rating"}.issubset(df.columns):
        st.error("The dataset needs UserId, AttractionId and Rating columns.")
    else:
        users = sorted(df["UserId"].dropna().unique())

        selected_user = st.selectbox(
            "Select User ID",
            users[:1000] if len(users) > 1000 else users
        )

        number = st.slider("Number of recommendations", 5, 20, 10)

        if st.button("Get Recommendations"):
            ratings = (
                df.groupby(["UserId", "AttractionId"])["Rating"]
                .mean()
                .reset_index()
            )

            visited = set(
                ratings.loc[
                    ratings["UserId"] == selected_user,
                    "AttractionId"
                ]
            )

            stats = (
                ratings.groupby("AttractionId")
                .agg(
                    AvgRating=("Rating", "mean"),
                    RatingCount=("Rating", "count")
                )
                .reset_index()
            )

            stats["Score"] = (
                stats["AvgRating"] *
                np.log1p(stats["RatingCount"])
            )

            recommendations = (
                stats[~stats["AttractionId"].isin(visited)]
                .sort_values("Score", ascending=False)
                .head(number)
            )

            if "Attraction" in df.columns:
                names = df[["AttractionId", "Attraction"]].drop_duplicates()
                recommendations = recommendations.merge(
                    names,
                    on="AttractionId",
                    how="left"
                )

            st.dataframe(recommendations, use_container_width=True)

st.divider()
st.caption("Tourism Experience Analytics | Machine Learning Project")
