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

try:
    import cv2
    st.success("✅ OpenCV imported successfully")
except Exception as e:
    st.error(f"❌ OpenCV failed: {e}")

try:
    from sklearn.cluster import KMeans
    st.success("✅ Scikit-learn imported successfully")
except Exception as e:
    st.error(f"❌ Scikit-learn failed: {e}")

try:
    import plotly.express as px
    st.success("✅ Plotly imported successfully")
except Exception as e:
    st.error(f"❌ Plotly failed: {e}")

# Basic file upload test
st.subheader("📄 File Upload Test")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"✅ Image loaded: {image.size} pixels, {image.mode} mode")

        # Test OpenCV conversion if available
        try:
            import cv2
            import numpy as np
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            st.success(f"✅ OpenCV processing successful - converted to grayscale ({gray.shape})")
        except Exception as e:
            st.warning(f"⚠️ OpenCV processing failed: {e}")

    except Exception as e:
        st.error(f"❌ Image loading failed: {e}")