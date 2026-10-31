"""
PowerPoint Presentation Generator for AKTU ML (BOE-073) Assessment Project
Creates a high-quality presentation: docs/Project_Presentation.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)  # 16:9 widescreen layout

    # Color Palette
    PRIMARY = RGBColor(26, 54, 93)     # Deep Navy
    SECONDARY = RGBColor(43, 108, 176) # Accent Blue
    TEXT_DARK = RGBColor(45, 55, 72)   # Charcoal
    TEXT_LIGHT = RGBColor(255, 255, 255)
    BG_LIGHT = RGBColor(247, 250, 252) # Soft off-white
    HIGHLIGHT = RGBColor(49, 130, 206)

    def add_header(slide, title_text, category_text="AKTU B.TECH 7TH SEMESTER | MACHINE LEARNING (BOE-073)"):
        # Header banner
        header_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = PRIMARY
        header_shape.line.fill.background()

        # Accent stripe
        stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.15), Inches(13.333), Inches(0.08))
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = SECONDARY
        stripe.line.fill.background()

        # Header Title
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(0.95))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0)
        
        p_sub = tf.paragraphs[0]
        p_sub.text = category_text.upper()
        p_sub.font.size = Pt(10)
        p_sub.font.bold = True
        p_sub.font.color.rgb = RGBColor(190, 227, 248)

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_LIGHT

    # ==================== SLIDE 1: Title Slide ====================
    slide_layout = prs.slide_layouts[6] # Blank
    s1 = prs.slides.add_slide(slide_layout)

    # Background
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRIMARY
    bg1.line.fill.background()

    # Title Card
    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(11), Inches(4.5))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow\nB.Tech 4th Year, 7th Semester | Open Elective - II"
    p0.font.size = Pt(14)
    p0.font.color.rgb = RGBColor(190, 227, 248)
    p0.font.bold = True
    p0.space_after = Pt(20)

    p1 = tf1.add_paragraph()
    p1.text = "Comparative Study and Empirical Evaluation of Classical Machine Learning Algorithms for Diagnostic Medical Classification"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT
    p1.space_after = Pt(25)

    p2 = tf1.add_paragraph()
    p2.text = "Course Code: BOE-073 / KOE-073: Machine Learning\nPrescribed Text: Machine Learning by Tom M. Mitchell (McGraw-Hill)\nProject Domain: Healthcare Analytics & Statistical Hypothesis Testing"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(226, 232, 240)

    # ==================== SLIDE 2: Problem Statement & Motivation ====================
    s2 = prs.slides.add_slide(slide_layout)
    add_header(s2, "1. Project Motivation & Clinical Objective")

    box2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf2 = box2.text_frame
    tf2.word_wrap = True

    def add_bullet(tf, title, body):
        p = tf.add_paragraph()
        p.text = f"•  {title}: "
        p.font.bold = True
        p.font.size = Pt(16)
        p.font.color.rgb = PRIMARY
        p.space_after = Pt(4)

        run = p.add_run()
        run.text = body
        run.font.bold = False
        run.font.size = Pt(15)
        run.font.color.rgb = TEXT_DARK
        p.space_after = Pt(18)

    add_bullet(tf2, "Clinical Importance", "Breast cancer remains one of the most diagnosed malignancies worldwide. Early, highly accurate differentiation between malignant (cancerous) and benign biopsies reduces unnecessary surgical interventions and saves lives.")
    add_bullet(tf2, "Academic Motivation", "Most student projects arbitrarily execute a single machine learning model. This project establishes a rigorous empirical benchmark comparing four major algorithmic paradigms prescribed in the AKTU syllabus.")
    add_bullet(tf2, "Core Objective", "Evaluate Decision Trees, Neural Networks, Naïve Bayes, and k-NN under strict statistical hypothesis testing (10-Fold Cross-Validation and Paired Student's t-tests) rather than misleading single-split metrics.")

    # ==================== SLIDE 3: Tom Mitchell (T, P, E) Formalization ====================
    s3 = prs.slides.add_slide(slide_layout)
    add_header(s3, "2. Tom M. Mitchell's Formal Problem Definition (T, P, E)")

    # 3 Cards for T, P, E
    cards = [
        ("TASK (T)", "Binary diagnostic classification of digitized fine needle aspirate (FNA) cell biopsy samples into Malignant (Class 0) or Benign (Class 1).", PRIMARY),
        ("PERFORMANCE (P)", "Classification Accuracy, Sensitivity/Recall, Precision, F1-Score, ROC-AUC, and Paired Student's t-test p-value across folds.", SECONDARY),
        ("EXPERIENCE (E)", "Supervised database of 569 historical patient biopsy records with 30 continuous dimensional cell nucleus measurements.", PRIMARY)
    ]

    for idx, (title, desc, color) in enumerate(cards):
        left = Inches(0.8 + idx * 3.95)
        c_shape = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.6), Inches(3.7), Inches(4.8))
        c_shape.fill.solid()
        c_shape.fill.fore_color.rgb = BG_LIGHT
        c_shape.line.color.rgb = color
        c_shape.line.width = Pt(2)

        # Card Header
        ch = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.6), Inches(3.7), Inches(0.9))
        ch.fill.solid()
        ch.fill.fore_color.rgb = color
        ch.line.fill.background()
        ch_tf = ch.text_frame
        ch_tf.margin_top = Inches(0.2)
        p = ch_tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

        # Card Body
        tb = s3.shapes.add_textbox(left + Inches(0.2), Inches(2.7), Inches(3.3), Inches(3.5))
        tf = tb.text_frame
        tf.word_wrap = True
        p_body = tf.paragraphs[0]
        p_body.text = desc
        p_body.font.size = Pt(14)
        p_body.font.color.rgb = TEXT_DARK

    # ==================== SLIDE 4: AKTU Syllabus Alignment ====================
    s4 = prs.slides.add_slide(slide_layout)
    add_header(s4, "3. Direct AKTU (BOE-073) Syllabus Alignment")

    units = [
        ("Unit II: Decision Tree Learning", "ID3 algorithm, Entropy, Information Gain, and tree pruning constraints (avoiding overfitting)."),
        ("Unit II: Artificial Neural Networks", "Perceptron learning, Delta Rule, Multi-Layer Perceptrons (MLP), and Error Backpropagation gradient descent."),
        ("Unit III: Bayesian Learning", "Bayes' theorem, Maximum A Posteriori (MAP), Gaussian Naïve Bayes under conditional independence assumption."),
        ("Unit III: Evaluating Hypotheses", "10-Fold Cross-Validation, sample variance, standard error, and Paired Student's t-test comparing learning algorithms."),
        ("Unit IV: Instance-Based Learning", "k-Nearest Neighbors (k-NN) lazy learning, Euclidean distance metric, and majority voting.")
    ]

    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))
    tf4 = tb4.text_frame
    tf4.word_wrap = True

    for u_title, u_desc in units:
        add_bullet(tf4, u_title, u_desc)

    # ==================== SLIDE 5: Modular Architecture ====================
    s5 = prs.slides.add_slide(slide_layout)
    add_header(s5, "4. Production-Grade Modular Repository Architecture")

    tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    add_bullet(tf5, "data/", "Holds clean CSV files: dataset.csv (full 569 rows), train_data.csv (80%), and test_data.csv (20%).")
    add_bullet(tf5, "src/", "Modular components: data_loader.py, models_sklearn.py, models_scratch.py, hypothesis_testing.py, visualizer.py.")
    add_bullet(tf5, "tests/", "Automated unittest suite (test_pipeline.py) and scratch validation (test_scratch_models.py).")

    tb5_r = s5.shapes.add_textbox(Inches(6.8), Inches(1.5), Inches(5.8), Inches(5.2))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True

    add_bullet(tf5_r, "notebooks/", "Diagnostic_Benchmark_Notebook.ipynb: step-by-step interactive Jupyter notebook.")
    add_bullet(tf5_r, "docs/", "Formal assessment report (PROJECT_REPORT.md) and presentation slides.")
    add_bullet(tf5_r, "outputs/", "Generated high-resolution plots: model_comparison.png, confusion_matrices.png, roc_curves.png, cv_boxplots.png.")

    # ==================== SLIDE 6: Experimental Benchmark Table ====================
    s6 = prs.slides.add_slide(slide_layout)
    add_header(s6, "5. Experimental Results: Model Performance Benchmark")

    # Add Table
    rows, cols = 5, 6
    left, top, width, height = Inches(0.8), Inches(1.6), Inches(11.7), Inches(3.2)
    table_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    tbl = table_shape.table

    headers = ["Machine Learning Algorithm", "Syllabus Unit", "Accuracy", "Precision", "Recall", "F1-Score"]
    data = [
        ["Decision Tree (ID3 / Entropy)", "Unit II", "93.86%", "95.77%", "94.44%", "95.10%"],
        ["Naive Bayes (Gaussian)", "Unit III", "92.98%", "94.44%", "94.44%", "94.44%"],
        ["k-Nearest Neighbors (k=5)", "Unit IV", "95.61%", "95.89%", "97.22%", "96.55%"],
        ["Artificial Neural Net (MLP)", "Unit II", "95.61%", "98.55%", "94.44%", "96.45%"]
    ]

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

    for row_idx, row in enumerate(data):
        for col_idx, val in enumerate(row):
            cell = tbl.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG_LIGHT if row_idx % 2 == 0 else RGBColor(237, 242, 247)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(13)
            p.font.color.rgb = TEXT_DARK
            p.alignment = PP_ALIGN.CENTER

    # Key observation box below table
    obs_box = s6.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8))
    obs_tf = obs_box.text_frame
    obs_tf.word_wrap = True
    add_bullet(obs_tf, "Key Finding", "k-NN achieved the highest Recall (97.22%), critically minimizing false negatives in clinical cancer diagnosis. Neural Networks (MLP) achieved the highest Precision (98.55%).")

    # ==================== SLIDE 7: Visual Benchmark (Image Embed) ====================
    s7 = prs.slides.add_slide(slide_layout)
    add_header(s7, "6. Visual Evaluation & Diagnostic Metrics")

    img_path1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "model_comparison.png")
    img_path2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "confusion_matrices.png")

    if os.path.exists(img_path1):
        s7.shapes.add_picture(img_path1, Inches(0.8), Inches(1.6), Inches(5.6), Inches(4.5))
    if os.path.exists(img_path2):
        s7.shapes.add_picture(img_path2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(4.5))

    # Caption
    c_box = s7.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.8))
    cp = c_box.text_frame.paragraphs[0]
    cp.text = "Left: Four-metric comparative bar chart across syllabus models. Right: 2x2 grid of confusion matrices on unseen test biopsy cases."
    cp.font.size = Pt(12)
    cp.font.italic = True
    cp.font.color.rgb = TEXT_DARK

    # ==================== SLIDE 8: Unit III Hypothesis Testing ====================
    s8 = prs.slides.add_slide(slide_layout)
    add_header(s8, "7. Unit III: Hypothesis Testing & Paired Student's t-Tests")

    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.5))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    add_bullet(tf8, "10-Fold Cross-Validation Variance Estimation", "Evaluates true generalization accuracy rather than single-split variance. Neural Network achieved 97.36% mean accuracy with 95% Confidence Interval [95.66%, 99.06%]. k-NN followed closely with 96.48%.")
    add_bullet(tf8, "Paired Student's t-Test (Comparing Learning Algorithms)", "Null Hypothesis (H0): Mean performance difference d_bar = 0 between two algorithms across all 10 folds at significance level alpha = 0.05.")
    add_bullet(tf8, "Decision Tree vs. Neural Network", "t-statistic = -3.50, p-value = 0.0067 < 0.05 -> Reject H0. The Neural Network's performance advantage over the Decision Tree is statistically significant.")
    add_bullet(tf8, "k-NN vs. Neural Network", "t-statistic = -0.89, p-value = 0.394 > 0.05 -> Fail to Reject H0. The performance difference between k-NN and Neural Network is NOT statistically significant.")

    # ==================== SLIDE 9: Scratch vs Scikit-Learn Verification ====================
    s9 = prs.slides.add_slide(slide_layout)
    add_header(s9, "8. Algorithmic Proof: From-Scratch Pure Python Models")

    tb9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf9 = tb9.text_frame
    tf9.word_wrap = True

    add_bullet(tf9, "Algorithmic Integrity (Tom Mitchell Math)", "Implemented pure mathematical algorithms from scratch using NumPy to demonstrate complete algorithmic understanding beyond high-level library calls.")
    add_bullet(tf9, "Scratch k-NN (Unit IV)", "Built with custom Euclidean distance, neighbor sorting, and majority voting. Achieved 100.00% exact prediction agreement with Scikit-Learn's KNeighborsClassifier.")
    add_bullet(tf9, "Scratch Gaussian Naïve Bayes (Unit III)", "Built with Maximum Likelihood Estimation of Gaussian mean, variance, and log-likelihood summation. Achieved 100.00% exact prediction agreement with Scikit-Learn's GaussianNB.")
    add_bullet(tf9, "Scratch Perceptron (Unit II)", "Built with the classic Delta rule weight update. Achieved 96.49% prediction agreement.")

    # ==================== SLIDE 10: Interactive Web Demo (Streamlit) ====================
    s10 = prs.slides.add_slide(slide_layout)
    add_header(s10, "9. Interactive Streamlit Web Application (Viva Demo)")

    tb10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    add_bullet(tf10, "Live Browser Dashboard", "Built with Streamlit (app.py) running on localhost:8501 for live interactive demonstration during viva evaluation.")
    add_bullet(tf10, "Real-Time Experimentation", "Sliders allow examiners to adjust test split ratios and algorithm hyperparameters (k in k-NN, tree depth in Decision Tree) in real-time.")
    add_bullet(tf10, "Live Diagnostic Predictor", "Allows the user or examiner to input continuous cell measurements (radius, texture, perimeter) and view live Malignant/Benign predictions with probability confidences.")

    # ==================== SLIDE 11: Conclusion & Key Takeaways ====================
    s11 = prs.slides.add_slide(slide_layout)
    add_header(s11, "10. Conclusions & Key Takeaways")

    tb11 = s11.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    add_bullet(tf11, "Continuous Feature Efficacy", "Distance-based (k-NN) and non-linear gradient (MLP) architectures outperformed axis-aligned Decision Trees on continuous diagnostic features.")
    add_bullet(tf11, "Statistical Rigor", "Demonstrated that high single-split test accuracy must be supported by 10-fold cross-validation and paired t-tests to verify statistical validity.")
    add_bullet(tf11, "Curriculum Mastery", "Successfully synthesized, implemented, verified, and packaged the core machine learning paradigms of AKTU BOE-073 into a production-grade academic project.")

    # ==================== SLIDE 12: Thank You / Viva Q&A ====================
    s12 = prs.slides.add_slide(slide_layout)
    bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = PRIMARY
    bg12.line.fill.background()

    tb12 = s12.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10.3), Inches(3.5))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    p_ty = tf12.paragraphs[0]
    p_ty.text = "Thank You!"
    p_ty.font.size = Pt(44)
    p_ty.font.bold = True
    p_ty.font.color.rgb = TEXT_LIGHT
    p_ty.alignment = PP_ALIGN.CENTER
    p_ty.space_after = Pt(20)

    p_qa = tf12.add_paragraph()
    p_qa.text = "Questions & Viva Voce Discussion\n\nAKTU B.Tech Machine Learning (BOE-073) Assessment"
    p_qa.font.size = Pt(20)
    p_qa.font.color.rgb = RGBColor(190, 227, 248)
    p_qa.alignment = PP_ALIGN.CENTER

    # Save Presentation
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    ppt_path = os.path.join(docs_dir, "Project_Presentation.pptx")
    prs.save(ppt_path)
    print(f"[SUCCESS] PowerPoint presentation saved to: {ppt_path}")

if __name__ == "__main__":
    create_presentation()

