# scanner_app.py

import streamlit as st
import numpy as np
import cv2
from scanner_utils import scan_document
from PIL import Image
import io

st.set_page_config(page_title="Document Scanner", layout="centered")
st.title("Document Scanner Web App")

uploaded_files = st.file_uploader("Upload one or more document images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"Original: {file.name}")
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        img_np = np.array(image)
        scanned = scan_document(img_np)

        if scanned is not None:
            st.subheader("Scanned Output")
            st.image(scanned, use_container_width=True)

            # Download button
            img_pil = Image.fromarray(scanned)
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format="PNG")
            st.download_button(label="Download Scanned Image", data=img_byte_arr.getvalue(), file_name=f"scanned_{file.name}", mime="image/png")
        else:
            st.warning("No document boundary detected!")