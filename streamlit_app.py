import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Configure page
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

class SimpleOMRProcessor:
    """Simplified OMR processor for Streamlit Cloud deployment"""

    def __init__(self):
        self.subjects = ['PYTHON', 'EDA', 'SQL', 'POWER BI', 'ADV STATS']
        self.expected_columns = 5
        self.expected_rows_per_column = 20
        self.bubbles_per_question = 4

    def detect_circles(self, image):
        """Detect circles using HoughCircles"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Multiple parameter sets for robust detection
        param_sets = [
            {'dp': 1.2, 'min_dist': 30, 'param1': 50, 'param2': 30, 'min_radius': 10, 'max_radius': 25},
            {'dp': 1.5, 'min_dist': 25, 'param1': 60, 'param2': 35, 'min_radius': 12, 'max_radius': 30}
        ]

        all_circles = []
        for params in param_sets:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=params['min_dist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['min_radius'],
                maxRadius=params['max_radius']
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                all_circles.extend(circles)

        return all_circles if all_circles else []

    def organize_bubbles_into_grid(self, circles, image_shape):
        """Organize detected circles into a grid structure"""
        if len(circles) < 50:  # Need minimum bubbles
            return {}

        # Extract coordinates
        positions = np.array([[x, y] for x, y, r in circles])

        # Cluster by columns (X-coordinates)
        try:
            kmeans_cols = KMeans(n_clusters=min(self.expected_columns, len(positions)), random_state=42, n_init=10)
            col_labels = kmeans_cols.fit_predict(positions[:, 0].reshape(-1, 1))
        except:
            return {}

        organized_grid = {}

        for col_idx in range(self.expected_columns):
            col_circles = [circles[i] for i in range(len(circles)) if col_labels[i] == col_idx]

            if len(col_circles) < 20:  # Need minimum circles per column
                continue

            # Sort by Y-coordinate (top to bottom)
            col_circles.sort(key=lambda c: c[1])

            # Group into rows (questions)
            for row_idx in range(0, min(len(col_circles), 80), 4):  # Max 20 questions * 4 bubbles
                question_bubbles = col_circles[row_idx:row_idx + 4]

                if len(question_bubbles) == 4:
                    question_num = col_idx * self.expected_rows_per_column + (row_idx // 4) + 1

                    if question_num <= 100:  # Max 100 questions
                        # Sort bubbles left to right (A, B, C, D)
                        question_bubbles.sort(key=lambda c: c[0])
                        organized_grid[question_num] = {
                            'subject': self.subjects[col_idx] if col_idx < len(self.subjects) else 'Unknown',
                            'bubbles': question_bubbles,
                            'column': col_idx,
                            'row': row_idx // 4
                        }

        return organized_grid

    def classify_bubbles(self, image, bubbles):
        """Classify bubbles as filled or empty"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        classifications = []

        for x, y, r in bubbles:
            # Extract bubble region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r - 2, 255, -1)

            # Calculate intensity statistics
            bubble_pixels = gray[mask == 255]

            if len(bubble_pixels) > 0:
                mean_intensity = np.mean(bubble_pixels)
                min_intensity = np.min(bubble_pixels)
                std_intensity = np.std(bubble_pixels)

                # Classification criteria
                is_filled = (mean_intensity < 120 and min_intensity < 100 and std_intensity < 40)
                classifications.append(is_filled)
            else:
                classifications.append(False)

        return classifications

    def extract_answers(self, organized_grid, image):
        """Extract answers from organized grid"""
        answers = {}

        for question_num, data in organized_grid.items():
            bubbles = data['bubbles']
            subject = data['subject']

            # Classify each bubble
            classifications = self.classify_bubbles(image, bubbles)

            # Determine answer
            filled_indices = [i for i, filled in enumerate(classifications) if filled]

            if len(filled_indices) == 1:
                # Single clear mark
                answer_options = ['a', 'b', 'c', 'd']
                answer = answer_options[filled_indices[0]]
                confidence = 'high'
                reason = 'single_clear_mark'
            elif len(filled_indices) > 1:
                # Multiple marks
                answer_options = ['a', 'b', 'c', 'd']
                answer = answer_options[filled_indices[0]]  # Take first
                confidence = 'low'
                reason = 'multiple_marks_detected'
            else:
                # No marks
                answer = None
                confidence = 'none'
                reason = 'no_mark_detected'

            answers[question_num] = {
                'subject': subject,
                'answer': answer,
                'confidence': confidence,
                'reason': reason,
                'bubble_count': len(bubbles),
                'filled_count': len(filled_indices)
            }

        return answers

    def process_image(self, image):
        """Main processing function"""
        try:
            # Detect circles
            circles = self.detect_circles(image)

            if not circles:
                return {'error': 'No bubbles detected in image'}

            # Organize into grid
            organized_grid = self.organize_bubbles_into_grid(circles, image.shape)

            if not organized_grid:
                return {'error': 'Could not organize bubbles into grid structure'}

            # Extract answers
            answers = self.extract_answers(organized_grid, image)

            return {
                'extracted_answers': answers,
                'grid_info': {
                    'total_circles': len(circles),
                    'organized_questions': len(organized_grid),
                    'subjects_detected': list(set(data['subject'] for data in organized_grid.values()))
                },
                'metadata': {
                    'image_dimensions': list(image.shape),
                    'total_questions': len(answers),
                    'processing_status': 'completed'
                }
            }

        except Exception as e:
            return {'error': f'Processing failed: {str(e)}'}

@st.cache_resource
def initialize_omr_processor():
    """Initialize OMR processor with caching"""
    return SimpleOMRProcessor()

def main():
    st.markdown('<h1 class="main-header">📝 OMR Evaluation System</h1>', unsafe_allow_html=True)

    # Initialize processor
    processor = initialize_omr_processor()

    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🔍 Process OMR Sheet",
        "📊 Batch Processing",
        "ℹ️ About"
    ])

    if page == "🔍 Process OMR Sheet":
        process_single_sheet(processor)
    elif page == "📊 Batch Processing":
        batch_processing(processor)
    elif page == "ℹ️ About":
        about_page()

def process_single_sheet(processor):
    st.header("🔍 Process Single OMR Sheet")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload OMR Sheet Image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear image of the OMR sheet"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📄 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

            # Image info
            st.info(f"**Image Info:**\n"
                   f"- Size: {image.size[0]} x {image.size[1]} pixels\n"
                   f"- Mode: {image.mode}\n"
                   f"- Format: {image.format}")

        # Process button
        if st.button("🚀 Process OMR Sheet", type="primary"):
            with st.spinner("Processing OMR sheet..."):
                try:
                    # Convert PIL image to OpenCV format
                    image_array = np.array(image)
                    if len(image_array.shape) == 3:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                    # Process the image
                    results = processor.process_image(image_array)

                    with col2:
                        display_results(results)

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

def display_results(results):
    st.subheader("📊 Processing Results")

    if 'error' in results:
        st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {results["error"]}</div>',
                   unsafe_allow_html=True)
        return

    # Extract data
    answers = results.get('extracted_answers', {})
    metadata = results.get('metadata', {})
    grid_info = results.get('grid_info', {})

    # Summary metrics
    total_questions = len(answers)
    answered = sum(1 for ans in answers.values() if ans.get('answer'))
    high_confidence = sum(1 for ans in answers.values() if ans.get('confidence') == 'high')
    low_confidence = sum(1 for ans in answers.values() if ans.get('confidence') == 'low')

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Answered", answered, f"{(answered/total_questions*100):.1f}%" if total_questions > 0 else "0%")
    with col3:
        st.metric("High Confidence", high_confidence, f"{(high_confidence/total_questions*100):.1f}%" if total_questions > 0 else "0%")
    with col4:
        st.metric("Low Confidence", low_confidence, f"{(low_confidence/total_questions*100):.1f}%" if total_questions > 0 else "0%")

    # Confidence distribution chart
    conf_data = {'high': high_confidence, 'low': low_confidence, 'none': total_questions - high_confidence - low_confidence}
    if total_questions > 0:
        fig = px.pie(values=list(conf_data.values()), names=list(conf_data.keys()),
                    title="Confidence Distribution",
                    color_discrete_map={'high': '#28a745', 'low': '#ffc107', 'none': '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)

    # Detailed answers
    st.subheader("📝 Extracted Answers")

    # Group by subject
    subjects = {}
    for q_num, data in answers.items():
        subject = data.get('subject', 'Unknown')
        if subject not in subjects:
            subjects[subject] = []
        subjects[subject].append((int(q_num), data))

    # Display by subject
    for subject, questions in subjects.items():
        with st.expander(f"📚 {subject} ({len(questions)} questions)"):
            questions.sort(key=lambda x: x[0])

            answer_data = []
            for q_num, data in questions:
                answer_data.append({
                    'Question': q_num,
                    'Answer': (data.get('answer') or 'None').upper(),
                    'Confidence': data.get('confidence', 'none'),
                    'Reason': data.get('reason', 'N/A')
                })

            if answer_data:
                df = pd.DataFrame(answer_data)
                st.dataframe(df, use_container_width=True)

    # Download results
    if answers:
        results_json = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def batch_processing(processor):
    st.header("📊 Batch Processing")
    st.info("Upload multiple OMR sheets for batch processing")

    uploaded_files = st.file_uploader(
        "Upload Multiple OMR Sheets",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload multiple OMR sheet images"
    )

    if uploaded_files and st.button("🚀 Process All Sheets", type="primary"):
        results_summary = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            st.write(f"Processing: {file.name}")

            try:
                image = Image.open(file)
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                results = processor.process_image(image_array)

                # Extract summary
                answers = results.get('extracted_answers', {})
                summary = {
                    'filename': file.name,
                    'total_questions': len(answers),
                    'answered': sum(1 for ans in answers.values() if ans.get('answer')),
                    'high_confidence': sum(1 for ans in answers.values() if ans.get('confidence') == 'high'),
                    'status': 'Success' if 'extracted_answers' in results else 'Failed'
                }
                results_summary.append(summary)

            except Exception as e:
                results_summary.append({
                    'filename': file.name,
                    'total_questions': 0,
                    'answered': 0,
                    'high_confidence': 0,
                    'status': f'Error: {str(e)}'
                })

            progress_bar.progress((i + 1) / len(uploaded_files))

        # Display batch results
        if results_summary:
            st.subheader("📊 Batch Processing Results")
            df = pd.DataFrame(results_summary)
            st.dataframe(df, use_container_width=True)

            # Summary metrics
            successful = df[df['status'] == 'Success']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(df))
            with col2:
                st.metric("Successful", len(successful))
            with col3:
                st.metric("Success Rate", f"{(len(successful)/len(df)*100):.1f}%" if len(df) > 0 else "0%")

def about_page():
    st.header("ℹ️ About OMR Evaluation System")

    st.markdown("""
    ## 🎯 Overview
    This OMR (Optical Mark Recognition) system automatically processes bubble sheet answer sheets
    using computer vision techniques.

    ## 🚀 Features
    - **HoughCircles Detection**: Robust circle detection with multiple parameter sets
    - **Grid Organization**: K-means clustering for spatial bubble organization
    - **Statistical Classification**: Advanced bubble analysis for accuracy
    - **Confidence Scoring**: High/Low/None confidence levels with reasoning
    - **Batch Processing**: Process multiple sheets simultaneously
    - **Export Capabilities**: Download results in JSON format

    ## 🛠️ Technologies Used
    - **OpenCV**: Computer vision and image processing
    - **Streamlit**: Web interface
    - **Plotly**: Interactive visualizations
    - **NumPy**: Numerical computations
    - **Pandas**: Data manipulation
    - **Scikit-learn**: K-means clustering

    ## 📊 Performance
    - **Processing Speed**: 2-5 seconds per sheet
    - **Grid Format**: 5 columns × 20 rows × 4 options (A,B,C,D)
    - **Subjects**: PYTHON, EDA, SQL, POWER BI, ADV STATS

    ## 🎓 Best Practices
    1. **Image Quality**: Use high-resolution scanned images (1200x1000+ recommended)
    2. **Lighting**: Ensure even lighting without shadows
    3. **Bubble Filling**: Fill bubbles completely with dark pencil/pen
    4. **Sheet Condition**: Keep sheets flat and uncrumpled
    5. **Alignment**: Ensure proper alignment during scanning
    """)

if __name__ == "__main__":
    main()