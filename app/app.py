import io
import os
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_jet_colormap():
    if hasattr(cm, "get_cmap"):
        return cm.get_cmap("jet")

    import matplotlib

    return matplotlib.colormaps["jet"]


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = get_jet_colormap()

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img
    )

    return superimposed_img


def pil_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_bar_chart_png(y_pred):
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.barh(
        class_names, y_pred, height=0.55, align="center", color="#4C78A8"
    )
    for i, (c, p) in enumerate(zip(class_names, y_pred)):
        ax.text(p + 2, i - 0.2, f"{p:.2f}%")
    ax.grid(axis="x")
    ax.set_xlim([0, 120])
    ax.set_xticks(range(0, 101, 20))
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def build_pdf_report(
    input_png, gradcam_png, chart_png, y_pred, grade, probability
):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Knee Osteoarthritis Analysis Report",
        author="Knee OA DL App",
    )

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    small = ParagraphStyle(
        "small", parent=normal, fontSize=9, textColor=colors.grey
    )

    elements = []
    elements.append(
        Paragraph("Knee Osteoarthritis Analysis Report", styles["Title"])
    )
    elements.append(
        Paragraph(
            f"Generated on {datetime.now():%Y-%m-%d %H:%M:%S}",
            small,
        )
    )
    elements.append(Spacer(1, 6 * mm))

    summary_data = [
        ["Predicted severity grade", f"{grade} (Grade {class_names.index(grade)})"],
        ["Confidence", f"{probability:.2f}%"],
        ["Model", "Xception (fine-tuned)"],
    ]
    summary = Table(summary_data, colWidths=[65 * mm, 105 * mm])
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0B0B0")),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(summary)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph("Images", styles["Heading2"]))
    img_input = RLImage(io.BytesIO(input_png), width=78 * mm, height=78 * mm)
    img_gradcam = RLImage(
        io.BytesIO(gradcam_png), width=78 * mm, height=78 * mm
    )
    images = Table(
        [
            [img_input, img_gradcam],
            [
                Paragraph("Input X-ray", small),
                Paragraph("Grad-CAM (model attention)", small),
            ],
        ],
        colWidths=[85 * mm, 85 * mm],
    )
    images.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(images)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph("Probability per grade", styles["Heading2"]))
    rows = [["Grade", "KL score", "Probability"]]
    for name, p in zip(class_names, y_pred):
        rows.append([name, str(class_names.index(name)), f"{p:.2f}%"])
    prob_table = Table(rows, colWidths=[60 * mm, 40 * mm, 70 * mm])
    prob_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C78A8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0B0B0")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(prob_table)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph("Probability chart", styles["Heading2"]))
    elements.append(
        RLImage(io.BytesIO(chart_png), width=170 * mm, height=68 * mm)
    )
    elements.append(Spacer(1, 6 * mm))

    elements.append(
        Paragraph(
            "Disclaimer: This report is generated by an automated deep "
            "learning model for educational and research purposes only. It is "
            "not a medical diagnosis. Always consult a qualified healthcare "
            "professional.",
            small,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


icon = Image.open("app/img/mdc.png")
st.set_page_config(
    page_title="Severity Analysis of Arthrosis in the Knee",
    page_icon=icon,
)

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

MODEL_PATH = "./src/models/model_Xception_ft.hdf5"


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return

    import gdown

    with st.spinner("Downloading the model (first run, ~239 MB)..."):
        gdown.download(
            "https://drive.google.com/uc?id=1vPS_j2AW3M1W8GydREEDw6CDENSgVwCy",
            MODEL_PATH,
            quiet=False,
        )


ensure_model()

model = tf.keras.models.load_model(MODEL_PATH)
target_size = (224, 224)

# Grad-CAM
grad_model = tf.keras.models.clone_model(model)
grad_model.set_weights(model.get_weights())
grad_model.layers[-1].activation = None
grad_model = tf.keras.models.Model(
    inputs=[grad_model.inputs],
    outputs=[
        grad_model.get_layer("global_average_pooling2d_1").input,
        grad_model.output,
    ],
)

# Sidebar
with st.sidebar:
    st.image(icon)
    st.subheader("Final Project - MDC013")
    st.caption("=== Fernanda Rodriguez ===")

    st.subheader(":arrow_up: Upload image")
    uploaded_file = st.file_uploader("Choose x-ray image")


# Body
st.header("Severity Analysis of Arthrosis in the Knee")

col1, col2 = st.columns(2)

if "result" not in st.session_state:
    st.session_state.result = None

if uploaded_file is not None:
    with col1:
        st.subheader(":camera: Input")
        st.image(uploaded_file, use_column_width=True)

        img = tf.keras.preprocessing.image.load_img(
            uploaded_file, target_size=target_size
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_aux = img.copy()

        if st.button(
            ":arrows_counterclockwise: Predict Arthrosis in the Knee"
        ):
            img_array = np.expand_dims(img_aux, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(
                img_array
            )

            with st.spinner("Wait for it..."):
                y_pred = model.predict(img_array)

            y_pred = 100 * y_pred[0]

            probability = float(np.amax(y_pred))
            number = np.where(y_pred == np.amax(y_pred))
            grade = str(class_names[np.amax(number)])

            heatmap = make_gradcam_heatmap(grad_model, img_array)
            gradcam_image = save_and_display_gradcam(img, heatmap)

            input_png = pil_to_png_bytes(
                tf.keras.preprocessing.image.array_to_img(img_aux)
            )
            gradcam_png = pil_to_png_bytes(gradcam_image)
            chart_png = make_bar_chart_png(y_pred)

            pdf_bytes = build_pdf_report(
                input_png,
                gradcam_png,
                chart_png,
                y_pred,
                grade,
                probability,
            )

            st.session_state.result = {
                "y_pred": y_pred.tolist(),
                "grade": grade,
                "probability": probability,
                "gradcam_png": gradcam_png,
                "chart_png": chart_png,
                "pdf": pdf_bytes,
                "time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
else:
    st.session_state.result = None

result = st.session_state.result

if result is not None:
    with col1:
        st.subheader(":white_check_mark: Prediction")
        st.metric(
            label="Severity Grade:",
            value=f"{result['grade']} - {result['probability']:.2f}%",
        )

    with col2:
        st.subheader(":mag: Explainability")
        st.image(io.BytesIO(result["gradcam_png"]), use_column_width=True)

        st.subheader(":bar_chart: Analysis")
        st.image(io.BytesIO(result["chart_png"]), use_column_width=True)

    st.download_button(
        label=":page_facing_up: Download PDF Report",
        data=result["pdf"],
        file_name=f"knee_oa_report_{result['time']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
