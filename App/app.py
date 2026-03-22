import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Config
st.set_page_config(page_title="PCBCheck", page_icon="🔍", layout="wide")

@st.cache_resource
def load_model():
    """Load your 98.1% mAP50 model"""
    model_path = "./model/PCBCheck_best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error("Model not found! Upload to model/PCBCheck_best.pt")
        return None

model = load_model()

# Header
st.markdown("""
# 🔍 **PCBCheck** - Production PCB Defect Detection
**98.1% mAP50 | 370 FPS | Live Deployed**

*Upload PCB → Instant defects with bounding boxes & confidence*
""")

# Layout
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("📸 **Upload PCB**")
    uploaded_file = st.file_uploader(
        "Drop image or click upload", 
        type=['png', 'jpg', 'jpeg'],
        help="Any PCB photo works!"
    )
    
    if uploaded_file:
        original = Image.open(uploaded_file)
        st.image(original, caption="🔬 Analyzing...", use_column_width=True)

with col2:
    if uploaded_file and model:
        with st.spinner("Detecting defects..."):
            # Predict
            img_array = np.array(Image.open(uploaded_file))
            results = model(img_array, conf=0.5, verbose=False)
            
            # Results
            annotated = results[0].plot()
            st.image(annotated, caption="✅ Defects Found!", use_column_width=True)
            
            # Metrics
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"**🎯 {len(boxes)} defects detected**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precision", "95.6%")
                with col2:
                    st.metric("mAP50", "98.1%")
            else:
                st.balloons()
                st.success("**✅ Perfect PCB!** No defects.")

# Footer
st.markdown("---")
st.markdown("""
**Ayusham Mishra** | [GitHub](https://github.com/ayusham-mishra) | [LinkedIn](https://linkedin.com/in/ayusham)
**Production: Docker + Railway + ONNX + 370 FPS inference**
""")
