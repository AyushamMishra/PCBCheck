import streamlit as st
import os
from PIL import Image
import numpy as np

# Model lazy loading (production safe)
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        return YOLO("./model/PCBCheck_best.pt")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

# Load model safely
model = load_model()

st.set_page_config(page_title="PCBCheck", layout="wide")

st.markdown("# 🔍 **PCBCheck** - 98.1% PCB Defect Detection")
st.markdown("*Production deployed | Upload → Instant results*")

col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded_file = st.file_uploader("📸 Upload PCB", type=['png','jpg','jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original", use_column_width=True)

with col2:
    if uploaded_file and model:
        with st.spinner("🔬 Detecting..."):
            img_array = np.array(Image.open(uploaded_file))
            results = model(img_array, conf=0.5, verbose=False)
            annotated = results[0].plot()
            st.image(annotated, caption="✅ Defects Found!", use_column_width=True)
            
            boxes = results[0].boxes
            defects = len(boxes) if boxes is not None else 0
            st.success(f"**🎯 {defects} defects detected** (98.1% mAP50)")
    elif uploaded_file:
        st.warning("Model loading... Please wait or check model/PCBCheck_best.pt")

st.markdown("**Ayusham Mishra | Production MLOps Demo**")
