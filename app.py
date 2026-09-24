from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# App configuration
# ============================================================

st.set_page_config(
    page_title="App User Behavior Segmentation",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

REQUIRED_FILES = {
    "segmented_users": OUTPUT_DIR / "segmented_users.csv",
    "cluster_performance": OUTPUT_DIR / "cluster_performance.csv",
    "segment_profile": OUTPUT_DIR / "final_segment_profile.csv",
    "distribution": OUTPUT_DIR / "segment_distribution.csv",
    "business_actions": OUTPUT_DIR / "business_action_mapping.csv",
    "summary": OUTPUT_DIR / "final_business_summary.csv",
    "pca": OUTPUT_DIR / "pca_coordinates.csv",
}


# ============================================================
# Helpers
# ============================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def require_file(path, label):
    if not path.exists():
        st.error(
            f"Required output is missing: `{path}`\n\n"
            f"Run the clean notebook completely before launching Streamlit."
        )
        st.stop()


def find_column(df, candidates):
    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None


def format_number(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


# ============================================================
# Load authoritative outputs
# ============================================================

for label, path in REQUIRED_FILES.items():
    require_file(path, label)

users = load_csv(REQUIRED_FILES["segmented_users"])
performance = load_csv(REQUIRED_FILES["cluster_performance"])
profile = load_csv(REQUIRED_FILES["segment_profile"])
distribution = load_csv(REQUIRED_FILES["distribution"])
actions = load_csv(REQUIRED_FILES["business_actions"])
summary = load_csv(REQUIRED_FILES["summary"])
pca = load_csv(REQUIRED_FILES["pca"])


# ============================================================
# Establish ONE authoritative cluster -> segment mapping
# ============================================================

cluster_col = find_column(performance, ["cluster"])
segment_col = find_column(performance, ["segment_name"])

if cluster_col is None or segment_col is None:
    st.error(
        "cluster_performance.csv must contain both `cluster` and "
        "`segment_name` columns."
    )
    st.stop()

mapping_df = performance[[cluster_col, segment_col]].copy()
mapping_df.columns = ["cluster", "segment_name"]

mapping_df["cluster"] = pd.to_numeric(
    mapping_df["cluster"], errors="coerce"
)

mapping_df = mapping_df.dropna(subset=["cluster", "segment_name"])
mapping_df["cluster"] = mapping_df["cluster"].astype(int)

if mapping_df["cluster"].duplicated().any():
    st.error("Duplicate cluster IDs found in cluster_performance.csv.")
    st.stop()

if mapping_df["segment_name"].duplicated().any():
    st.error("Duplicate segment names found in cluster_performance.csv.")
    st.stop()

EXPECTED_SEGMENTS = {
    "High Users",
    "Moderate Users",
    "Low Users",
    "Occasional Users",
}

if set(mapping_df["segment_name"]) != EXPECTED_SEGMENTS:
    st.warning(
        "The notebook output does not currently contain exactly the expected "
        "four business segment names."
    )

CLUSTER_TO_SEGMENT = dict(
    zip(mapping_df["cluster"], mapping_df["segment_name"])
)

SEGMENT_TO_CLUSTER = {
    segment: cluster
    for cluster, segment in CLUSTER_TO_SEGMENT.items()
}

SEGMENT_ORDER = [
    "High Users",
    "Moderate Users",
    "Low Users",
    "Occasional Users",
]


# ============================================================
# Normalize customer data using authoritative mapping
# ============================================================

user_cluster_col = find_column(users, ["cluster"])
user_segment_col = find_column(users, ["segment_name"])

if user_cluster_col is None:
    st.error("segmented_users.csv is missing the cluster column.")
    st.stop()

users[user_cluster_col] = pd.to_numeric(
    users[user_cluster_col], errors="coerce"
)

# IMPORTANT:
# Ignore any stale segment_name already present in segmented_users.csv.
# Recreate it from the authoritative cluster_performance mapping.
users["segment_name"] = users[user_cluster_col].map(CLUSTER_TO_SEGMENT)

if users["segment_name"].isna().any():
    st.error(
        "Some users have a cluster ID that does not exist in "
        "cluster_performance.csv."
    )
    st.stop()


# ============================================================
# Normalize profile
# ============================================================

profile_cluster_col = find_column(profile, ["cluster"])
profile_segment_col = find_column(profile, ["segment_name"])

if profile_cluster_col is not None:
    profile[profile_cluster_col] = pd.to_numeric(
        profile[profile_cluster_col], errors="coerce"
    )

if profile_cluster_col is not None:
    # Ignore stale profile segment labels and recreate them.
    profile["segment_name"] = profile[profile_cluster_col].map(
        CLUSTER_TO_SEGMENT
    )


# ============================================================
# Normalize distribution
# ============================================================

distribution_segment_col = find_column(
    distribution, ["segment_name"]
)

if distribution_segment_col is not None:
    distribution["segment_name"] = distribution[
        distribution_segment_col
    ]


# ============================================================
# Normalize business actions
# ============================================================

action_segment_col = find_column(actions, ["segment_name"])

if action_segment_col is None:
    st.error(
        "business_action_mapping.csv is missing segment_name."
    )
    st.stop()

actions["segment_name"] = actions[action_segment_col]


# ============================================================
# Header
# ============================================================

st.title("📊 App User Behavior Segmentation")
st.caption(
    "K-Means based behavioral segmentation with customer-level profiling "
    "and business action mapping."
)

st.info(
    "Segment names are dynamically derived from cluster performance in "
    "`cluster_performance.csv`. Cluster numbers are technical IDs and are "
    "not used as business segment names."
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Segment Analysis",
        "Customer Explorer",
        "Customer Targeting",
        "Model Insights",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Current Mapping")

for cluster_id in sorted(CLUSTER_TO_SEGMENT):
    st.sidebar.write(
        f"Cluster {cluster_id} → {CLUSTER_TO_SEGMENT[cluster_id]}"
    )


# ============================================================
# Dashboard
# ============================================================

if page == "Dashboard":

    st.header("Dashboard Overview")

    total_users = len(users)
    total_segments = users["segment_name"].nunique()

    high_count = int(
        (users["segment_name"] == "High Users").sum()
    )

    low_count = int(
        (users["segment_name"] == "Low Users").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Users", f"{total_users:,}")
    c2.metric("Segments", total_segments)
    c3.metric("High Users", f"{high_count:,}")
    c4.metric("Low Users", f"{low_count:,}")

    st.subheader("Segment Distribution")

    dist = (
        users["segment_name"]
        .value_counts()
        .reindex(SEGMENT_ORDER)
        .fillna(0)
        .astype(int)
    )

    distribution_display = pd.DataFrame({
        "Segment": dist.index,
        "Users": dist.values,
        "Percentage": (
            dist.values / total_users * 100
        ).round(2),
    })

    st.dataframe(
        distribution_display,
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        distribution_display["Segment"],
        distribution_display["Users"],
    )
    ax.set_title("Users by Business Segment")
    ax.set_ylabel("Number of Users")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dynamic Cluster Mapping")

    st.dataframe(
        mapping_df.sort_values("cluster"),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Segment Analysis
# ============================================================

elif page == "Segment Analysis":

    st.header("Segment Analysis")
    st.write(
        "Understand the behavioral characteristics of each business segment."
    )

    selected_segment = st.selectbox(
        "Select Segment",
        SEGMENT_ORDER,
    )

    selected_cluster = SEGMENT_TO_CLUSTER.get(selected_segment)

    st.subheader(selected_segment)

    st.write(
        f"Technical K-Means cluster: **Cluster {selected_cluster}**"
    )

    # Always retrieve performance by cluster from the authoritative file.
    perf_row = performance[
        performance[cluster_col] == selected_cluster
    ]

    if perf_row.empty:
        st.error("No performance record exists for this cluster.")
        st.stop()

    perf = perf_row.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    activity_col = find_column(
        performance, ["activity_score"]
    )
    value_col = find_column(
        performance, ["value_score"]
    )
    retention_col = find_column(
        performance, ["retention_score"]
    )
    overall_col = find_column(
        performance, ["overall_behavior_score"]
    )

    c1.metric(
        "Activity Score",
        format_number(perf[activity_col])
        if activity_col else "N/A",
    )
    c2.metric(
        "Value Score",
        format_number(perf[value_col])
        if value_col else "N/A",
    )
    c3.metric(
        "Retention Score",
        format_number(perf[retention_col])
        if retention_col else "N/A",
    )
    c4.metric(
        "Overall Behavior",
        format_number(perf[overall_col])
        if overall_col else "N/A",
    )

    selected_users = users[
        users["segment_name"] == selected_segment
    ]

    st.subheader("Customer Count")

    st.metric(
        "Users in Segment",
        f"{len(selected_users):,}",
    )

    st.subheader("Behavioral Profile")

    # Select the row from profile using cluster, not segment text.
    profile_row = pd.DataFrame()

    if profile_cluster_col is not None:
        profile_row = profile[
            profile[profile_cluster_col] == selected_cluster
        ]

    if not profile_row.empty:
        profile_display = profile_row.drop(
            columns=[
                c for c in [
                    profile_cluster_col,
                ]
                if c is not None and c in profile_row.columns
            ],
            errors="ignore",
        )

        profile_display = profile_display.drop(
            columns=["segment_name"],
            errors="ignore",
        )

        st.dataframe(
            profile_display.T.rename(columns={profile_display.index[0]: "Value"})
            if len(profile_display) == 1
            else profile_display,
            use_container_width=True,
        )
    else:
        st.warning("No profile record found for this cluster.")

    st.subheader("Business Interpretation")

    action_row = actions[
        actions["segment_name"] == selected_segment
    ]

    if not action_row.empty:
        action = action_row.iloc[0]

        for column in action.index:
            if column == "segment_name":
                continue

            label = str(column).replace("_", " ").title()

            st.markdown(f"**{label}**")
            st.write(action[column])
    else:
        st.warning("No business action mapping found.")


# ============================================================
# Customer Explorer
# ============================================================

elif page == "Customer Explorer":

    st.header("Customer Explorer")
    st.write(
        "Identify the exact users belonging to each behavioral segment."
    )

    selected_segment = st.selectbox(
        "Select Segment",
        SEGMENT_ORDER,
    )

    selected_users = users[
        users["segment_name"] == selected_segment
    ].copy()

    st.metric(
        "Users in Segment",
        f"{len(selected_users):,}",
    )

    search_text = st.text_input(
        "Search by User ID",
        "",
    )

    user_id_col = find_column(
        selected_users,
        [
            "user_id",
            "userid",
            "user id",
            "customer_id",
            "customerid",
            "customer id",
        ],
    )

    if search_text and user_id_col:
        selected_users = selected_users[
            selected_users[user_id_col]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False,
            )
        ]

    st.dataframe(
        selected_users,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Customer Targeting
# ============================================================

elif page == "Customer Targeting":

    st.header("Customer Targeting")
    st.write(
        "Translate segment behavior into customer-level business actions."
    )

    selected_segment = st.selectbox(
        "Select Target Segment",
        SEGMENT_ORDER,
    )

    target_users = users[
        users["segment_name"] == selected_segment
    ].copy()

    action_row = actions[
        actions["segment_name"] == selected_segment
    ]

    st.subheader("Target Audience")

    st.metric(
        "Customers Available for Targeting",
        f"{len(target_users):,}",
    )

    if not action_row.empty:
        action = action_row.iloc[0]

        characteristics_col = find_column(
            action_row,
            [
                "customer_characteristics",
                "characteristics",
                "customer_profile",
            ],
        )

        value_col = find_column(
            action_row,
            [
                "business_value",
                "value",
            ],
        )

        recommendation_col = find_column(
            action_row,
            [
                "recommended_action",
                "recommended_actions",
                "action",
            ],
        )

        if characteristics_col:
            st.subheader("Customer Characteristics")
            st.write(action[characteristics_col])

        if value_col:
            st.subheader("Business Value")
            st.write(action[value_col])

        if recommendation_col:
            st.subheader("Recommended Action")
            st.write(action[recommendation_col])

    st.subheader("Target Customers")

    user_id_col = find_column(
        target_users,
        [
            "user_id",
            "userid",
            "user id",
            "customer_id",
            "customerid",
            "customer id",
        ],
    )

    if user_id_col:
        st.dataframe(
            target_users[[user_id_col, "cluster", "segment_name"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            target_users,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Model Insights
# ============================================================

elif page == "Model Insights":

    st.header("Model Insights")

    st.subheader("Cluster Performance")

    display_performance = performance.copy()

    st.dataframe(
        display_performance.sort_values("cluster"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("PCA Visualization")

    pca_x = find_column(
        pca,
        ["pca_1", "PC1", "principal_component_1"],
    )
    pca_y = find_column(
        pca,
        ["pca_2", "PC2", "principal_component_2"],
    )
    pca_segment = find_column(
        pca,
        ["segment_name"],
    )

    if pca_x and pca_y:

        if pca_segment is not None:
            # Re-map PCA segment labels through cluster so old labels
            # cannot contaminate the visualization.
            pca_cluster = find_column(
                pca,
                ["cluster"],
            )

            if pca_cluster:
                pca["display_segment"] = pd.to_numeric(
                    pca[pca_cluster],
                    errors="coerce",
                ).map(CLUSTER_TO_SEGMENT)
            else:
                pca["display_segment"] = pca[pca_segment]
        else:
            pca_cluster = find_column(
                pca,
                ["cluster"],
            )

            if pca_cluster:
                pca["display_segment"] = pd.to_numeric(
                    pca[pca_cluster],
                    errors="coerce",
                ).map(CLUSTER_TO_SEGMENT)

        fig, ax = plt.subplots(figsize=(10, 7))

        for segment in SEGMENT_ORDER:
            if "display_segment" not in pca.columns:
                continue

            subset = pca[
                pca["display_segment"] == segment
            ]

            ax.scatter(
                subset[pca_x],
                subset[pca_y],
                s=10,
                alpha=0.4,
                label=segment,
            )

        ax.set_title("PCA – User Behavioral Segments")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning(
            "PCA coordinate columns were not found in pca_coordinates.csv."
        )

    st.subheader("Cluster-to-Segment Mapping")

    st.dataframe(
        mapping_df.sort_values("cluster"),
        use_container_width=True,
        hide_index=True,
    )
