import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="App User Behavior Segmentation",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "app_user_behavior_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    pca = joblib.load(MODEL_DIR / "pca.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans_model.pkl")

    with open(MODEL_DIR / "feature_names.txt", "r", encoding="utf-8") as file:
        features = [line.strip() for line in file.readlines()]

    return scaler, pca, kmeans, features


@st.cache_data
def load_outputs():
    cluster_profiles = pd.read_csv(
        OUTPUT_DIR / "cluster_profiles.csv",
        index_col=0
    )

    segment_profiles = pd.read_csv(
        OUTPUT_DIR / "segment_profiles.csv",
        index_col=0
    )

    user_assignments = pd.read_csv(
        OUTPUT_DIR / "user_cluster_assignments.csv"
    )

    segment_sizes = pd.read_csv(
        OUTPUT_DIR / "segment_sizes.csv"
    )

    business_actions = pd.read_csv(
        OUTPUT_DIR / "business_action_mapping.csv"
    )

    return (
        cluster_profiles,
        segment_profiles,
        user_assignments,
        segment_sizes,
        business_actions
    )


# ---------------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------------

try:
    df = load_data()

    scaler, pca, kmeans, behavior_features = load_models()

    (
        cluster_profiles,
        segment_profiles,
        user_assignments,
        segment_sizes,
        business_actions
    ) = load_outputs()

    # Merge the ML cluster and segment results with the original dataset
    df = df.merge(
        user_assignments[["user_id", "cluster", "segment"]],
        on="user_id",
        how="left"
    )

except Exception as e:
    st.error("Unable to load the project files.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📊 App User Behavior Segmentation")

st.markdown(
    """
    ### Unsupervised Machine Learning Dashboard

    This application segments app users based on their behavioral
    and engagement patterns using **K-Means clustering**.
    """
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Segment Analysis",
        "User Search",
        "At-Risk Users",
        "PCA Visualization"
    ]
)


# ---------------------------------------------------------
# COMMON METRICS
# ---------------------------------------------------------

total_users = len(df)
total_segments = df["segment"].nunique()

average_engagement = df["engagement_score"].mean()
average_churn = df["churn_risk_score"].mean()


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

if page == "Overview":

    st.header("📌 Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Users",
        f"{total_users:,}"
    )

    col2.metric(
        "User Segments",
        total_segments
    )

    col3.metric(
        "Avg. Engagement Score",
        f"{average_engagement:.2f}"
    )

    col4.metric(
        "Avg. Churn Risk",
        f"{average_churn:.2f}"
    )

    st.divider()

    st.subheader("User Segment Distribution")

    distribution = (
        df["segment"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "Segment",
        "Users"
    ]

    st.bar_chart(
        distribution.set_index("Segment")
    )

    st.subheader("Segment Summary")

    summary = (
        df.groupby("segment")
        .agg(
            Users=("user_id", "count"),
            Avg_Sessions=("sessions_per_week", "mean"),
            Avg_Session_Duration=(
                "avg_session_duration_min",
                "mean"
            ),
            Avg_Engagement=(
                "engagement_score",
                "mean"
            ),
            Avg_Churn_Risk=(
                "churn_risk_score",
                "mean"
            )
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# SEGMENT ANALYSIS
# ---------------------------------------------------------

elif page == "Segment Analysis":

    st.header("🔎 Segment Analysis")

    segments = sorted(
        df["segment"].dropna().unique()
    )

    selected_segment = st.selectbox(
        "Select a user segment",
        segments
    )

    segment_data = df[
        df["segment"] == selected_segment
    ]

    st.subheader(
        f"{selected_segment}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Users",
        f"{len(segment_data):,}"
    )

    col2.metric(
        "Avg. Sessions / Week",
        f"{segment_data['sessions_per_week'].mean():.2f}"
    )

    col3.metric(
        "Avg. Engagement",
        f"{segment_data['engagement_score'].mean():.2f}"
    )

    col4.metric(
        "Avg. Churn Risk",
        f"{segment_data['churn_risk_score'].mean():.2f}"
    )

    st.divider()

    st.subheader("Behavioral Profile")

    profile_metrics = pd.DataFrame({
        "Metric": [
            "Average Session Duration",
            "Daily Active Minutes",
            "Feature Clicks / Session",
            "Notifications Opened / Week",
            "Pages Viewed / Session",
            "Content Downloads",
            "Social Shares",
            "Days Since Last Login"
        ],
        "Average": [
            segment_data["avg_session_duration_min"].mean(),
            segment_data["daily_active_minutes"].mean(),
            segment_data["feature_clicks_per_session"].mean(),
            segment_data["notifications_opened_per_week"].mean(),
            segment_data["pages_viewed_per_session"].mean(),
            segment_data["content_downloads"].mean(),
            segment_data["social_shares"].mean(),
            segment_data["days_since_last_login"].mean()
        ]
    })

    profile_metrics["Average"] = (
        profile_metrics["Average"].round(2)
    )

    st.dataframe(
        profile_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    action = business_actions[
        business_actions["segment"] == selected_segment
    ]

    if not action.empty:

        st.subheader("💡 Recommended Business Action")

        st.info(
            action.iloc[0]["recommended_actions"]
        )


# ---------------------------------------------------------
# USER SEARCH
# ---------------------------------------------------------

elif page == "User Search":

    st.header("👤 User Search")

    user_ids = df["user_id"].astype(str).tolist()

    selected_user = st.selectbox(
        "Select User ID",
        user_ids
    )

    user_data = df[
        df["user_id"].astype(str) == selected_user
    ]

    if not user_data.empty:

        user = user_data.iloc[0]

        st.subheader("User Profile")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Segment",
            user["segment"]
        )

        col2.metric(
            "Engagement Score",
            f"{user['engagement_score']:.2f}"
        )

        col3.metric(
            "Churn Risk",
            f"{user['churn_risk_score']:.2f}"
        )

        col4.metric(
            "Sessions / Week",
            f"{user['sessions_per_week']:.2f}"
        )

        st.divider()

        st.subheader("Behavioral Details")

        user_metrics = pd.DataFrame({
            "Metric": [
                "Average Session Duration",
                "Daily Active Minutes",
                "Feature Clicks / Session",
                "Notifications Opened / Week",
                "In-App Searches",
                "Pages Viewed / Session",
                "Crash Events",
                "Support Tickets",
                "Days Since Last Login",
                "Ads Clicked",
                "Content Downloads",
                "Social Shares"
            ],
            "Value": [
                user["avg_session_duration_min"],
                user["daily_active_minutes"],
                user["feature_clicks_per_session"],
                user["notifications_opened_per_week"],
                user["in_app_search_count"],
                user["pages_viewed_per_session"],
                user["crash_events_last_30_days"],
                user["support_tickets_raised"],
                user["days_since_last_login"],
                user["ads_clicked_last_30_days"],
                user["content_downloads"],
                user["social_shares"]
            ]
        })

        st.dataframe(
            user_metrics,
            use_container_width=True,
            hide_index=True
        )

        action = business_actions[
            business_actions["segment"] == user["segment"]
        ]

        if not action.empty:

            st.subheader("🎯 Recommended Action")

            st.success(
                action.iloc[0]["recommended_actions"]
            )


# ---------------------------------------------------------
# AT-RISK USERS
# ---------------------------------------------------------

elif page == "At-Risk Users":

    st.header("⚠️ At-Risk Users")

    at_risk_segment = "Low Engagement / At-Risk Users"

    at_risk_users = df[
        df["segment"] == at_risk_segment
    ].copy()

    if at_risk_users.empty:

        st.warning(
            "No users were assigned to the At-Risk segment."
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "At-Risk Users",
            f"{len(at_risk_users):,}"
        )

        col2.metric(
            "Avg. Engagement",
            f"{at_risk_users['engagement_score'].mean():.2f}"
        )

        col3.metric(
            "Avg. Churn Risk",
            f"{at_risk_users['churn_risk_score'].mean():.2f}"
        )

        st.divider()

        display_columns = [
            "user_id",
            "sessions_per_week",
            "avg_session_duration_min",
            "daily_active_minutes",
            "days_since_last_login",
            "engagement_score",
            "churn_risk_score"
        ]

        display_data = at_risk_users[
            display_columns
        ].sort_values(
            "churn_risk_score",
            ascending=False
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Recommended strategy: prioritize these users "
            "for retention offers, reminders and re-engagement campaigns."
        )


# ---------------------------------------------------------
# PCA VISUALIZATION
# ---------------------------------------------------------

elif page == "PCA Visualization":

    st.header("📈 PCA Cluster Visualization")

    st.write(
        """
        PCA reduces the behavioral feature space to two dimensions.
        Each point represents a user and the color represents their
        assigned behavioral segment.
        """
    )

    # Transform the original feature data
    X_app = df[behavior_features].copy()

    # Handle any missing values consistently
    X_app = X_app.fillna(
        X_app.median(numeric_only=True)
    )

    X_scaled_app = scaler.transform(X_app)

    X_pca_app = pca.transform(X_scaled_app)

    pca_display = pd.DataFrame({
        "PC1": X_pca_app[:, 0],
        "PC2": X_pca_app[:, 1],
        "Segment": df["segment"].values
    })

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    for segment in pca_display["Segment"].unique():

        subset = pca_display[
            pca_display["Segment"] == segment
        ]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            alpha=0.35,
            s=15,
            label=segment
        )

    ax.set_title(
        "User Segmentation Using PCA"
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    ax.legend()

    st.pyplot(fig)

    st.divider()

    st.subheader("PCA Explained Variance")

    variance_data = pd.DataFrame({
        "Component": [
            "PC1",
            "PC2"
        ],
        "Explained Variance": [
            pca.explained_variance_ratio_[0],
            pca.explained_variance_ratio_[1]
        ]
    })

    variance_data["Explained Variance"] = (
        variance_data["Explained Variance"] * 100
    ).round(2)

    st.dataframe(
        variance_data,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "App User Behavior Segmentation | "
    "Unsupervised Machine Learning"
)