App User Behavior Segmentation

Project Overview

App User Behavior Segmentation is an unsupervised machine learning project that analyzes application user behavior and groups users with similar behavioral patterns.

The project uses K-Means clustering to segment users based on behavioral characteristics such as activity, engagement, value-related behavior, and retention-related indicators.

The final business-facing segments are:

High Users

Moderate Users

Low Users

Occasional Users

The project also provides a Streamlit dashboard for interactive segment analysis, customer identification, targeting, and model insights.

Project Objective

The objective of this project is to:

Identify distinct user behavior patterns.

Group users based on behavioral similarity.

Profile each behavioral segment.

Identify the customers belonging to each segment.

Map segments to suitable business actions.

Provide an interactive dashboard for business analysis.

Dataset

The project uses an app user behavior dataset containing 50,000 users with behavioral information such as:

User information

Device details

Session activity

Engagement metrics

Interaction behavior

Retention/churn-related indicators

The clustering process is unsupervised and does not require a labeled target variable.

Note: The dataset CSV is expected to be placed manually in the data folder.

Expected dataset name:

app_user_behavior_dataset.csv

Project Approach

The notebook follows the complete project workflow:

1. Data Collection

Loads the app user behavior dataset from the data folder.

2. Data Understanding

Examines:

Dataset dimensions

Column names

Data types

Missing values

Duplicate records

Statistical summaries

3. Data Cleaning

Performs:

Duplicate removal

Numeric conversion where appropriate

Missing-value imputation

Infinite-value handling

Categorical-value handling

4. Feature Selection

Selects relevant behavioral variables representing:

Activity

Engagement

Value

Retention

Identifiers and non-behavioral fields are excluded from clustering.

5. Data Scaling

Uses StandardScaler to normalize the selected behavioral features.

6. Clustering Model Selection

Evaluates different K-Means cluster counts using:

Inertia

Elbow Method

Silhouette Score

7. Optimal Cluster Identification

The final project uses K = 4 to create four behavioral groups.

8. Cluster Assignment

K-Means assigns every user to one of four technical clusters.

9. Cluster-Level User Identification & Profiling

Analyzes each cluster using:

User count

Percentage of users

Behavioral averages

Activity score

Value score

Retention score

Overall behavior score

10. Business Insight Generation & Customer Action Mapping

Maps the behavioral groups to business-facing segments and recommended actions.

Segment Mapping

K-Means cluster numbers are technical identifiers and do not inherently represent business segment names.

The project determines the segment names from observed cluster performance.

Current mapping produced by the model:

Cluster

Business Segment

Cluster 0

High Users

Cluster 1

Low Users

Cluster 2

Moderate Users

Cluster 3

Occasional Users

The authoritative mapping is generated in:

outputs/cluster_performance.csv

The Streamlit application reads this mapping rather than hard-coding cluster numbers.

Business Segments

High Users

Users showing strong overall behavioral performance and engagement.

Potential business actions include:

Loyalty programs

Premium offers

Exclusive features

Personalized recommendations

Moderate Users

Users demonstrating consistent but moderate application activity.

Potential business actions include:

Personalized recommendations

Feature discovery

Targeted incentives

Engagement campaigns

Low Users

Users showing comparatively low activity and weaker retention-related behavior.

Potential business actions include:

Re-engagement campaigns

Win-back offers

Personalized notifications

Retention campaigns

Occasional Users

Users who interact with the application irregularly or infrequently.

Potential business actions include:

Usage reminders

Personalized content

Relevant promotions

Engagement campaigns

PCA Visualization

Principal Component Analysis (PCA) is used to reduce the scaled behavioral feature space to two dimensions for visualization.

The PCA visualization helps inspect the distribution and separation of the identified user clusters.

Streamlit Dashboard

The project includes an interactive Streamlit application with the following sections:

Dashboard

Displays:

Total users

Number of segments

High-user count

Low-user count

Segment distribution

Cluster-to-segment mapping

Segment Analysis

Allows users to select a business segment and view:

Activity Score

Value Score

Retention Score

Overall Behavior Score

Customer count

Behavioral profile

Business interpretation

Customer Explorer

Allows users to:

Select a segment

View customers within the segment

Search by User ID

Customer Targeting

Displays:

Target audience size

Customer characteristics

Business value

Recommended action

Target customers

Model Insights

Displays:

Cluster performance

PCA visualization

Cluster-to-segment mapping

Project Structure

app-user-behavior-segmentation/
│
├── data/
│   └── app_user_behavior_dataset.csv
│
├── notebooks/
│   └── app_user_behavior_segmentation_clean.ipynb
│
├── models/
│   ├── kmeans_model.joblib
│   ├── scaler.joblib
│   ├── pca_model.joblib
│   ├── cluster_features.txt
│   └── segment_mapping.txt
│
├── outputs/
│   ├── segmented_users.csv
│   ├── cluster_performance.csv
│   ├── final_segment_profile.csv
│   ├── segment_distribution.csv
│   ├── business_action_mapping.csv
│   ├── final_business_summary.csv
│   ├── high_users.csv
│   ├── moderate_users.csv
│   ├── low_users.csv
│   ├── occasional_users.csv
│   ├── clustering_metrics.csv
│   └── pca_coordinates.csv
│
├── app.py
├── requirements.txt
└── README.md

Installation

1. Clone or download the project

Open the project folder in VS Code.

2. Create a virtual environment

python -m venv venv

3. Activate the virtual environment

Windows:

venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

Running the Notebook

Open the notebook located in:

notebooks/app_user_behavior_segmentation_clean.ipynb

Run the notebook from the first cell to the final cell.

The notebook will:

Load the dataset.

Clean the data.

Train the K-Means model.

Generate cluster profiles.

Determine business segment names.

Generate customer-level assignments.

Create PCA output.

Save reporting outputs.

Save model artifacts.

Running the Streamlit Dashboard

After successfully running the notebook, make sure the required files exist in the outputs and models folders.

From the project root:

streamlit run app.py

Streamlit will start the dashboard locally.

Output Files

segmented_users.csv

Contains customer-level cluster and segment assignments.

cluster_performance.csv

Contains cluster-level activity, value, retention, overall behavior scores, and business segment names.

final_segment_profile.csv

Contains behavioral averages and profile information for each segment.

segment_distribution.csv

Contains segment-level user counts and percentages.

business_action_mapping.csv

Contains business interpretation and recommended actions for each segment.

final_business_summary.csv

Combines segment profiles and business actions into a reporting dataset.

Individual segment files

The project also creates:

high_users.csv
moderate_users.csv
low_users.csv
occasional_users.csv

These contain the users belonging to each business segment.

Model Artifacts

The following reusable model artifacts are saved in the models folder:

kmeans_model.joblib

scaler.joblib

pca_model.joblib

cluster_features.txt

segment_mapping.txt

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Joblib

Jupyter Notebook

Streamlit

Final Outcome

The project provides an end-to-end user segmentation solution:

User Behavior Data
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
K-Means Clustering
        ↓
4 Behavioral Clusters
        ↓
Cluster Performance Analysis
        ↓
Business Segment Mapping
        ↓
Customer-Level Identification
        ↓
Business Action Mapping
        ↓
Streamlit Dashboard

The solution converts raw behavioral data into customer segments that can be analyzed and used for targeted business actions.