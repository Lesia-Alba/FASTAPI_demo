import requests
import streamlit as st
from PIL import Image

API_URL = st.secrets.get(
    "API_URL",
    "https://5gidbl8icyoejn-8000.proxy.runpod.net"
)

st.set_page_config(page_title="BERT + YOLO11 Demo", layout="centered")
st.title("BERT + YOLO11 Demo")

task = st.radio("Choose a task", ["Text classification", "Brain MRI detection"])

if task == "Text classification":
    st.subheader("Text sentiment classification")
    text = st.text_area("Enter text", height=150)

    if st.button("Predict text"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/clf_text",
                    json={"text": text},
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction completed")
                    st.write("Label:", result["label"])
                    st.write("Confidence:", result.get("prob"))
                else:
                    st.error(f"Error: {response.status_code}")
                    st.json(response.json())
            except Exception as e:
                st.error(f"Request failed: {e}")

else:
    st.subheader("Brain MRI image detection")
    uploaded_file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Detect tumor"):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                response = requests.post(
                    f"{API_URL}/clf_image",
                    files=files,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("Detection completed")

                    detections = result.get("detections", [])

                    if detections:
                        for i, det in enumerate(detections, 1):
                            st.write(f"Detection {i}")
                            st.write("Class:", det["class_name"])
                            st.write("Confidence:", det["confidence"])
                            st.write("BBox:", det["bbox"])
                            st.write("---")
                    else:
                        st.info("No tumor-related objects detected.")

                    st.json(result)

                else:
                    st.error(f"Error: {response.status_code}")
                    st.json(response.json())

            except Exception as e:
                st.error(f"Request failed: {e}")