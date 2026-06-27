"""Generate project report PDF for CV capstone submission."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).parent / "docs" / "Face_Mask_Classifier_Report.pdf"


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "Face Mask Classification - CV Capstone Project", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def h1(self, text: str):
        self.set_font("Helvetica", "B", 14)
        self.ln(4)
        self.multi_cell(0, 8, text)
        self.ln(2)

    def h2(self, text: str):
        self.set_font("Helvetica", "B", 12)
        self.ln(3)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def body(self, text: str):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, f"  - {text}")


def build():
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.h1("Face Mask Classification Using MobileNetV2")
    pdf.body(
        "Author: Zach Aye\n"
        "Course: Computer Vision Capstone\n"
        "GitHub: https://github.com/Zachrist321/facemask_classifier_mbnv2\n"
        "Live App: https://facemask-classifier-mbnv2.streamlit.app\n"
        "Dataset: MAKS Face Mask Dataset (Kaggle)"
    )

    pdf.h1("1. Introduction")
    pdf.body(
        "Public health monitoring during respiratory disease outbreaks requires automated tools to "
        "detect whether people wear face masks correctly. Manual inspection does not scale in "
        "crowded environments such as hospitals, airports, and retail spaces.\n\n"
        "This project builds a deep learning image classification system that identifies mask "
        "compliance from face images. We use transfer learning with MobileNetV2 pretrained on "
        "ImageNet, fine-tuned on the MAKS dataset. The final deliverable is a Streamlit web "
        "application that accepts uploaded photos or webcam input and returns a class prediction "
        "with confidence scores."
    )

    pdf.h1("2. Problem Statement")
    pdf.body(
        "Given a face image, the system will classify mask compliance into three categories so "
        "that public-health and security teams can monitor mask usage in real time.\n\n"
        "Classes:\n"
        "  1. with_mask - face covered correctly\n"
        "  2. mask_weared_incorrect - mask present but worn incorrectly\n"
        "  3. without_mask - no mask detected"
    )

    pdf.h1("3. Methodology")
    pdf.h2("3.1 Dataset")
    pdf.body(
        "Source: MAKS (Kaggle) - 853 images with Pascal VOC XML bounding-box annotations.\n"
        "Preprocessing: Each annotated face region was cropped with 15% margin and resized to "
        "224x224 pixels, yielding 4,072 face samples.\n"
        "Split: 80/20 stratified per class (seed=42). Train: 3,259 | Validation: 813."
    )
    pdf.h2("3.2 Model")
    pdf.body(
        "Architecture: MobileNetV2 (ImageNet weights) with frozen backbone.\n"
        "Trainable head: GlobalAveragePooling -> Dropout(0.3) -> Dense(3, softmax).\n"
        "Only the classification head was trained (~0.3% of parameters) for fast GPU training."
    )
    pdf.h2("3.3 Training")
    pdf.body(
        "Platform: Google Colab (T4 GPU).\n"
        "Optimizer: Adam, lr=1e-3 | Batch size: 32 | Epochs: 15 (early stopping).\n"
        "Augmentation: random flip, brightness, contrast.\n"
        "Best validation accuracy: 92.5%."
    )
    pdf.h2("3.4 Evaluation Metrics")
    pdf.body("Accuracy, per-class Precision/Recall/F1-score, and confusion matrix on validation set.")

    pdf.h1("4. Proposed Solution")
    pdf.body(
        "A transfer-learning pipeline: (1) parse VOC annotations and crop faces, (2) train a "
        "MobileNetV2 classifier with frozen backbone, (3) evaluate on held-out validation crops, "
        "(4) deploy via Streamlit for upload/webcam inference."
    )

    pdf.h1("5. System Architecture")
    pdf.body(
        "Components:\n"
        "  - Data layer: MAKS images + XML annotations -> face crops\n"
        "  - Model layer: MobileNetV2 + custom softmax head\n"
        "  - Inference layer: TensorFlow prediction on 224x224 RGB input\n"
        "  - UI layer: Streamlit app (upload or webcam)\n"
        "  - Hosting: Streamlit Community Cloud linked to GitHub repository"
    )

    pdf.h1("6. Implementation Details")
    pdf.body(
        "Languages/Libraries: Python 3.11, TensorFlow 2.15, Streamlit, Pillow, scikit-learn.\n"
        "Hardware (training): Google Colab T4 GPU.\n"
        "Hardware (deployment): Streamlit Cloud CPU.\n"
        "Repository structure: streamlit_app.py (UI), src/ (data & model utilities), "
        "face_mask_artifacts/ (trained weights), notebooks/ (Colab training notebook)."
    )

    pdf.h1("7. Results")
    pdf.h2("7.1 Overall Performance")
    pdf.body(
        "Validation accuracy: 92.5%\n"
        "Weighted F1-score: 0.91"
    )
    pdf.h2("7.2 Per-Class Results")
    pdf.body(
        "with_mask:           Precision 0.94 | Recall 0.97 | F1 0.96 (646 samples)\n"
        "without_mask:        Precision 0.86 | Recall 0.85 | F1 0.85 (143 samples)\n"
        "mask_weared_incorrect: Precision 0.50 | Recall 0.08 | F1 0.14 (24 samples)"
    )
    pdf.h2("7.3 Failure Cases")
    pdf.body(
        "Four of five documented failures involved mask_weared_incorrect predicted as with_mask. "
        "Causes: (1) class imbalance - only 99 training samples for incorrect mask vs 2,586 for "
        "with_mask; (2) visually ambiguous partial-mask faces; (3) one without_mask case where "
        "shadow/hand near face resembled a mask edge."
    )
    pdf.h2("7.4 Limitations")
    pdf.body(
        "Model trained on cropped faces - full-scene photos without clear faces perform poorly. "
        "Incorrect-mask class needs more training data. iPhone HEIC files require conversion "
        "before upload."
    )

    pdf.h1("8. Deployment")
    pdf.body(
        "The application is deployed on Streamlit Community Cloud:\n"
        "  https://facemask-classifier-mbnv2.streamlit.app\n\n"
        "Source code:\n"
        "  https://github.com/Zachrist321/facemask_classifier_mbnv2\n\n"
        "Users upload a face photo or use a webcam. The app displays the predicted class and "
        "probability bars for all three classes."
    )

    pdf.h1("9. Conclusion")
    pdf.body(
        "We built a working face mask classifier achieving 92.5% validation accuracy using "
        "MobileNetV2 transfer learning. The system correctly identifies with_mask and without_mask "
        "in most cases. The mask_weared_incorrect class remains challenging due to limited data. "
        "Future work: collect more incorrect-mask samples, add face detection for full-scene "
        "images, and fine-tune the last MobileNet blocks."
    )

    pdf.h1("10. References")
    pdf.body(
        "[1] MAKS Dataset - Kaggle: Face Mask Detection Dataset\n"
        "[2] Howard et al., MobileNetV2, arXiv:1801.04381\n"
        "[3] TensorFlow Documentation - Transfer Learning\n"
        "[4] Streamlit Documentation - Community Cloud Deployment\n"
        "[5] Project II Guidelines - DNN Based Computer Vision Capstone"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
