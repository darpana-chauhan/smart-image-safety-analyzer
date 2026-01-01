import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Image Safety Analyzer",
    layout="centered"
)

st.title("🛡️ Smart Image Safety Analyzer")
st.write("Upload an image to analyze real-world safety risks")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded_image = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

# ---------------- CAPTION FUNCTION ----------------
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_length=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.lower()

# ---------------- SAFETY ANALYSIS (FIXED LOGIC) ----------------
def analyze_safety(caption):
    risks = []
    status = "✅ Safe"

    has_bike = "bike" in caption or "motorcycle" in caption
    has_helmet = "helmet" in caption

    has_fire = "fire" in caption
    has_smoke = "smoke" in caption
    has_construction = "construction" in caption

    # 🚲 Bike rules
    if has_bike and not has_helmet:
        status = "⚠️ Unsafe"
        risks.append("Wear a helmet while riding a bike.")

    if has_bike and has_helmet:
        risks.append("Good job wearing a helmet while riding.")

    # 🔥 Fire & smoke rules
    if has_fire:
        status = "⚠️ Unsafe"
        risks.append("Maintain a safe distance from fire.")

    if has_smoke:
        status = "⚠️ Unsafe"
        risks.append("Smoke can be harmful to health.")

    # 🏗️ Construction rule
    if has_construction:
        status = "⚠️ Unsafe"
        risks.append("Wear proper safety equipment at construction sites.")

    if not risks:
        risks.append("No major safety risk detected.")

    return status, risks

# ---------------- DISPLAY ----------------
if uploaded_image:
    image = Image.open(uploaded_image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True   # ✅ deprecated warning fixed
    )

    if st.button("Analyze Safety"):
        with st.spinner("Analyzing image safety..."):
            caption = generate_caption(image)
            status, advice = analyze_safety(caption)

        st.subheader("🧠 AI Description")
        st.write(caption)

        st.subheader("🚦 Safety Status")
        st.write(status)

        st.subheader("🛑 Safety Advice")
        for a in advice:
            st.write("•", a)



'''streamlit run app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.68.69:8501'''