import streamlit as st

st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📝",
    layout="wide"
)

st.title("📝 OMR Evaluation System")

# Test basic dependencies
st.subheader("🔍 Dependency Check")

try:
    import numpy as np
    st.success("✅ NumPy imported successfully")
except Exception as e:
    st.error(f"❌ NumPy failed: {e}")

try:
    import pandas as pd
    st.success("✅ Pandas imported successfully")
except Exception as e:
    st.error(f"❌ Pandas failed: {e}")

try:
    from PIL import Image
    st.success("✅ Pillow imported successfully")
except Exception as e:
    st.error(f"❌ Pillow failed: {e}")

# Basic file upload test
st.subheader("📄 File Upload Test")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"✅ Image loaded: {image.size} pixels, {image.mode} mode")
    except Exception as e:
        st.error(f"❌ Image loading failed: {e}")