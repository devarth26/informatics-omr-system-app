import streamlit as st

# Configure page
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide"
)

def main():
    st.title("📝 OMR Evaluation System")
    st.write("Testing basic deployment...")

    # Test import
    try:
        import pandas as pd
        import numpy as np
        st.success("✅ NumPy and Pandas imported successfully")
    except Exception as e:
        st.error(f"❌ NumPy/Pandas import failed: {e}")

    # Test OpenCV
    try:
        import cv2
        st.success("✅ OpenCV imported successfully")
    except Exception as e:
        st.error(f"❌ OpenCV import failed: {e}")

    # Test sklearn
    try:
        from sklearn.cluster import KMeans
        st.success("✅ Scikit-learn imported successfully")
    except Exception as e:
        st.error(f"❌ Scikit-learn import failed: {e}")

    # Test Plotly
    try:
        import plotly.express as px
        st.success("✅ Plotly imported successfully")
    except Exception as e:
        st.error(f"❌ Plotly import failed: {e}")

if __name__ == "__main__":
    main()