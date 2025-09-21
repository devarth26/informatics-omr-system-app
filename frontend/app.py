import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
from datetime import datetime
import sys
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, {"status": "unavailable"}

def get_system_info():
    """Get system information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/system-info", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def process_single_omr(file, set_letter, student_name=None, student_id=None):
    """Process a single OMR sheet"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"set_letter": set_letter}

        if student_name:
            data["student_name"] = student_name
        if student_id:
            data["student_id"] = student_id

        response = requests.post(
            f"{API_BASE_URL}/upload-and-process",
            files=files,
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def get_results(limit=50, offset=0):
    """Get processing results"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/results",
            params={"limit": limit, "offset": offset},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_statistics():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/statistics", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def download_csv():
    """Download CSV export"""
    try:
        response = requests.get(f"{API_BASE_URL}/export/csv", timeout=30)
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<h1 class="main-header">📝 OMR Evaluation & Scoring System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Check API health
    api_healthy, health_data = check_api_health()

    if not api_healthy:
        st.error("🚨 Backend API is not running. Please start the FastAPI server first.")
        st.info("Run: `cd backend && uvicorn main:app --reload` to start the API server")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload & Process", "Results Dashboard", "Statistics", "System Status"]
    )

    # Display system status in sidebar
    system_info = get_system_info()
    if system_info:
        st.sidebar.markdown("### System Status")
        if system_info.get('system_ready'):
            st.sidebar.success("✅ System Ready")
        else:
            st.sidebar.error("❌ System Not Ready")

    # Page routing
    if page == "Upload & Process":
        upload_page()
    elif page == "Results Dashboard":
        results_page()
    elif page == "Statistics":
        statistics_page()
    elif page == "System Status":
        system_status_page()

def upload_page():
    """Upload and process OMR sheets page"""
    st.header("📤 Upload OMR Sheets")

    # Processing options
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Single Sheet Processing")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose OMR sheet image",
            type=['jpg', 'jpeg', 'png'],
            key="single_upload"
        )

        if uploaded_file:
            # Display image
            st.image(uploaded_file, caption="Uploaded OMR Sheet", use_column_width=True)

            # Input fields
            set_letter = st.selectbox("Answer Set", ["A", "B"], key="single_set")
            student_name = st.text_input("Student Name (Optional)", key="single_name")
            student_id = st.text_input("Student ID (Optional)", key="single_id")

            if st.button("🔍 Process OMR Sheet", key="single_process"):
                with st.spinner("Processing OMR sheet..."):
                    success, result = process_single_omr(
                        uploaded_file, set_letter, student_name, student_id
                    )

                if success:
                    st.success("✅ Processing completed successfully!")

                    # Display results
                    result_data = result['result']

                    # Score summary
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Score", f"{result_data['total_score']}/100")
                    with col_b:
                        st.metric("Percentage", f"{result_data['percentage']:.1f}%")
                    with col_c:
                        st.metric("Quality Score", f"{result_data.get('quality_score', 0):.1f}%")

                    # Subject-wise scores
                    st.subheader("Subject-wise Performance")
                    subject_scores = result_data['subject_scores']

                    # Create bar chart
                    subjects = list(subject_scores.keys())
                    scores = [subject_scores[subj]['score'] for subj in subjects]
                    percentages = [subject_scores[subj]['percentage'] for subj in subjects]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=subjects,
                        y=scores,
                        text=[f"{score}/20<br>{perc:.1f}%" for score, perc in zip(scores, percentages)],
                        textposition='auto',
                        marker_color='lightblue'
                    ))
                    fig.update_layout(
                        title="Subject-wise Scores",
                        xaxis_title="Subjects",
                        yaxis_title="Score (out of 20)",
                        yaxis=dict(range=[0, 20])
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed results
                    if st.expander("📊 Detailed Results"):
                        st.json(result_data)

                    # Flags and warnings
                    if result_data.get('flags'):
                        st.warning("⚠️ Quality Flags:")
                        for flag in result_data['flags']:
                            st.write(f"• {flag}")

                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

    with col2:
        st.subheader("Batch Processing")

        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose multiple OMR sheet images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} files selected")

            set_letter_batch = st.selectbox("Answer Set", ["A", "B"], key="batch_set")

            if st.button("🔍 Process All Sheets", key="batch_process"):
                st.warning("🚧 Batch processing feature coming soon!")
                st.info("Currently, please process sheets one by one using the single processing option.")

def results_page():
    """Results dashboard page"""
    st.header("📊 Results Dashboard")

    # Get results
    results_data = get_results(limit=100)

    if not results_data or not results_data['results']:
        st.info("No results found. Process some OMR sheets first!")
        return

    results = results_data['results']

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sheets", len(df))
    with col2:
        avg_score = df['percentage'].mean()
        st.metric("Average Score", f"{avg_score:.1f}%")
    with col3:
        high_scorers = len(df[df['percentage'] >= 80])
        st.metric("High Scorers (≥80%)", high_scorers)
    with col4:
        if 'quality_score' in df.columns:
            avg_quality = df['quality_score'].mean()
            st.metric("Avg Quality", f"{avg_quality:.1f}%")

    # Score distribution chart
    st.subheader("Score Distribution")

    fig = px.histogram(
        df, x='percentage',
        bins=20,
        title="Distribution of Scores",
        labels={'percentage': 'Percentage Score', 'count': 'Number of Students'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Subject-wise performance
    if all(col in df.columns for col in ['python_score', 'eda_score', 'sql_score', 'powerbi_score', 'stats_score']):
        st.subheader("Subject-wise Performance")

        subject_cols = ['python_score', 'eda_score', 'sql_score', 'powerbi_score', 'stats_score']
        subject_means = [df[col].mean() for col in subject_cols]
        subject_names = ['Python', 'EDA', 'SQL', 'Power BI', 'Statistics']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=subject_names,
            y=subject_means,
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title="Average Scores by Subject",
            xaxis_title="Subjects",
            yaxis_title="Average Score (out of 20)",
            yaxis=dict(range=[0, 20])
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent results table
    st.subheader("Recent Results")

    # Format the dataframe for display
    display_cols = ['student_name', 'student_id', 'set_letter', 'total_score', 'percentage', 'processing_timestamp']
    display_df = df[display_cols].copy() if all(col in df.columns for col in display_cols) else df

    # Sort by timestamp if available
    if 'processing_timestamp' in display_df.columns:
        display_df = display_df.sort_values('processing_timestamp', ascending=False)

    st.dataframe(display_df.head(20), use_container_width=True)

    # Export functionality
    if st.button("📥 Download Results as CSV"):
        csv_data = download_csv()
        if csv_data:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
        else:
            st.error("Failed to generate CSV export")

def statistics_page():
    """Statistics page"""
    st.header("📈 System Statistics")

    stats_data = get_statistics()

    if not stats_data:
        st.error("Unable to fetch statistics")
        return

    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Results", stats_data.get('total_results', 0))
    with col2:
        avg_score = stats_data.get('average_score', 0)
        st.metric("Average Score", f"{avg_score:.1f}")
    with col3:
        st.metric("Min Score", stats_data.get('min_score', 0))
    with col4:
        st.metric("Max Score", stats_data.get('max_score', 0))

    # Subject statistics
    if 'subject_statistics' in stats_data:
        st.subheader("Subject-wise Statistics")

        subject_stats = stats_data['subject_statistics']
        if subject_stats:
            # Create comparison chart
            subjects = list(subject_stats.keys())
            means = [subject_stats[subj]['mean'] for subj in subjects]
            mins = [subject_stats[subj]['min'] for subj in subjects]
            maxs = [subject_stats[subj]['max'] for subj in subjects]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Average', x=subjects, y=means))
            fig.add_trace(go.Bar(name='Minimum', x=subjects, y=mins))
            fig.add_trace(go.Bar(name='Maximum', x=subjects, y=maxs))

            fig.update_layout(
                title="Subject Performance Statistics",
                xaxis_title="Subjects",
                yaxis_title="Score",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    if 'score_distribution' in stats_data:
        st.subheader("Score Distribution")

        distribution = stats_data['score_distribution']

        labels = ['Excellent (90-100%)', 'Good (80-89%)', 'Average (60-79%)', 'Below Average (40-59%)', 'Poor (<40%)']
        values = [
            distribution.get('excellent', 0),
            distribution.get('good', 0),
            distribution.get('average', 0),
            distribution.get('below_average', 0),
            distribution.get('poor', 0)
        ]

        fig = px.pie(values=values, names=labels, title="Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

def system_status_page():
    """System status page"""
    st.header("🔧 System Status")

    # API Health
    api_healthy, health_data = check_api_health()

    if api_healthy:
        st.success("✅ API Server is running")
        st.json(health_data)
    else:
        st.error("❌ API Server is not responding")

    # System information
    system_info = get_system_info()

    if system_info:
        st.subheader("System Information")

        processing_stats = system_info.get('processing_stats', {})
        database_stats = system_info.get('database_stats', {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Processing Engine")
            st.json(processing_stats)

        with col2:
            st.markdown("### Database")
            st.json(database_stats)

        # System readiness
        if system_info.get('system_ready'):
            st.success("🟢 System is ready for processing")
        else:
            st.error("🔴 System is not ready")

    # Configuration
    st.subheader("Configuration")
    st.info(f"API Base URL: {API_BASE_URL}")

    # Help section
    st.subheader("Help & Documentation")
    st.markdown("""
    ### How to Use the OMR System

    1. **Upload OMR Sheets**: Go to 'Upload & Process' tab and upload image files
    2. **Select Answer Set**: Choose the correct answer set (A or B)
    3. **Add Student Info**: Optionally add student name and ID
    4. **Process**: Click the process button to analyze the OMR sheet
    5. **View Results**: Check the results dashboard for all processed sheets

    ### Supported File Formats
    - JPEG (.jpg, .jpeg)
    - PNG (.png)

    ### System Requirements
    - Images should be clear and well-lit
    - OMR sheets should be properly aligned
    - Bubbles should be clearly marked with dark marks
    """)

if __name__ == "__main__":
    main()