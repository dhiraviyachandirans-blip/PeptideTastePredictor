

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Core computational helpers
from peptide_core import compute_features, compute_peptide_properties, validate_sequence, VALID_AA

# NEW imports for properties and sequence conversion
from Bio.SeqUtils import seq3
import py3Dmol


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="PepTastePredictor", layout="wide", page_icon="🧬")

# --- Custom CSS for a cleaner, more accessible look (with dark-mode support) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    /* Respect the user's OS preference by default (when not forcing theme) */
    @media (prefers-color-scheme: dark) {
        :root {
            --accent: #66b8ff;
            --accent-strong: #9ad0ff;
            --success: #2dd4bf;
            --warning: #ffb86b;
            --muted: #bfc7d6;

            --bg: #0b1220;
            --page-bg: linear-gradient(180deg, #071422 0%, #071428 100%);
            --card-bg: #0f1724;
            --text: #e6eef8;
            --card-shadow: rgba(2,6,23,0.6);
            --muted-contrast: #9aa5b4;
            --prob-bg: rgba(255,255,255,0.06);
        }
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); background: var(--page-bg); }

    .stApp { background: var(--page-bg); }

    .hero {
        padding: 18px;
        border-radius: 12px;
        background: linear-gradient(90deg,var(--card-bg), rgba(0,0,0,0.03));
        box-shadow: 0 6px 18px var(--card-shadow);
        color: var(--text);
    }

    .card {
        padding: 12px;
        border-radius: 10px;
        background: var(--card-bg);
        color: var(--text);
        box-shadow: 0 6px 18px var(--card-shadow);
    }

    .prob-bar { height: 12px; border-radius: 6px; }

    /* Accent and high-contrast text */
    .accent { color: var(--accent); font-weight:700 }
    .muted { color: var(--muted-contrast); }
    .small { font-size:0.95rem }

    /* Adjust progress bar backgrounds for theme */
    .card .prob-bg { background: var(--prob-bg); }

    /* High-contrast variant: stronger card contrast and accent tweaks */
    .variant-high, .dark.variant-high {
        --card-bg: #091018;
        --card-shadow: rgba(0,0,0,0.8);
        --accent: #9be7ff;
        --accent-strong: #c0f2ff;
        --text: #f0f6ff;
        --prob-bg: rgba(255,255,255,0.03);
    }

    /* Focus outlines for keyboard accessibility */
    :focus { outline: 3px solid rgba(13,110,253,0.12); outline-offset: 2px; }

    /* Screen-reader helper */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        border: 0;
    }

    /* Responsive tweaks */
    @media (max-width: 640px) {
      .hero { padding: 12px; }
      .card { padding: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Sidebar ---
with st.sidebar:
    st.image("static/logo.svg", width=140, caption="PepTastePredictor")
    st.markdown("# **PepTastePredictor**")
    st.markdown("Predict peptide taste classes, explore model metrics, and batch-predict from files.")

    # Theme selector: Auto follows OS setting; Light/Dark force explicit mode
    st.divider()
    # Read any existing query params to pre-populate the controls
    # Use the non-experimental API to read query params
    _qp = st.query_params
    _init_theme = _qp.get("theme", [None])[0] if "theme" in _qp else None
    _init_variant = _qp.get("variant", [None])[0] if "variant" in _qp else None
    _init_rem = (_qp.get("remember", ["0"])[0] == "1") if "remember" in _qp else False

    # Map query param to selectbox index
    if _init_theme == "dark":
        _theme_idx = 2
    elif _init_theme == "light":
        _theme_idx = 1
    else:
        _theme_idx = 0

    theme_choice = st.selectbox("Theme", ["Auto (system)", "Light", "Dark"], index=_theme_idx, key="theme_choice")
    st.caption("Theme controls: choose 'Auto' to follow your OS preference.")

    # Color variant selector for optional tweaks (Default / High contrast)
    _variant_idx = 1 if _init_variant == "high" else 0
    color_variant = st.selectbox("Color palette", ["Default", "High contrast"], index=_variant_idx, key="color_variant")

    # Remember option — stores selection in the URL query params
    remember_theme = st.checkbox("Remember my choice (store in URL)", value=_init_rem, key="remember_theme")

    st.divider()
    st.markdown("**Try a sample sequence**")
    sample_sequences = ["EDEGEQPRPF", "GGGSSH", "ACDEFGHIK", "FLGFR"]
    sample = st.selectbox("Sample sequences", sample_sequences)
    if st.button("Use sample"):
        # Place the sample directly into the main input via session state
        st.session_state["user_seq"] = sample

    st.divider()
    st.markdown("**Run**")
    st.caption("Run locally: streamlit run PeptideTaste.py")

    st.markdown("---")
    with st.expander("About / Accessibility", expanded=False):
        st.write(
            "This demo is optimized for readability: high contrast text, "
            "keyboard focus outlines, and clear labels."
        )
        st.write("Tip: Use the sample sequences to try predictions quickly.")
    st.markdown("**Notes**")
    st.write("- Upload files with a `peptide` column for batch predictions.")
    st.write("- ColabFold recommended for 3D structure generation.")

# Apply theme choice (Light/Dark/Auto) by toggling a .dark class on the document element.
# 'Auto (system)' leaves styling to CSS's prefers-color-scheme; 'Dark' forces dark; 'Light' forces light.
theme = st.session_state.get("theme_choice", "Auto (system)")
variant = st.session_state.get("color_variant", "Default")
force = "dark" if theme == "Dark" else ("light" if theme == "Light" else "auto")
variant_flag = "high" if variant == "High contrast" else "default"

# Persist in the URL query params only when requested, and only if values differ (avoids unnecessary reruns)
_qp = st.query_params
_desired = {"theme": ("dark" if force == "dark" else ("light" if force == "light" else "auto")),
            "variant": ("high" if variant_flag == "high" else "default")}
if st.session_state.get("remember_theme", False):
    _desired["remember"] = "1"
    # Compare carefully because st.query_params returns dict[str, list[str]]
    current_theme = _qp.get("theme", [None])[0] if "theme" in _qp else None
    current_variant = _qp.get("variant", [None])[0] if "variant" in _qp else None
    current_rem = _qp.get("remember", [None])[0] if "remember" in _qp else None
    if (current_theme != _desired["theme"] or current_variant != _desired["variant"] or current_rem != _desired["remember"]):
        st.experimental_set_query_params(**_desired)
else:
    # If the user opted to stop remembering and there is an existing remember marker, clear params
    if "remember" in _qp:
        st.experimental_set_query_params()

# Inject JS to toggle the relevant classes for theme and variant
components.html(f"""
<script>
(function() {{
    const mode = "{force}";
    const variant = "{variant_flag}";
    if (mode === "dark") {{
        document.documentElement.classList.add("dark");
    }} else if (mode === "light") {{
        document.documentElement.classList.remove("dark");
        document.documentElement.classList.add("light");
    }} else {{
        // Auto: remove explicit classes to let the browser decide
        document.documentElement.classList.remove("dark");
        document.documentElement.classList.remove("light");
    }}

    if (variant === "high") {{
        document.documentElement.classList.add("variant-high");
    }} else {{
        document.documentElement.classList.remove("variant-high");
    }}
}})();
</script>
""", height=0)

# --- Hero header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        """
        <div class='hero'>
          <h1 style='margin:0'>PepTastePredictor</h1>
          <p class='muted small' style='margin:0'>
          Predict peptide taste (Sweet / Salty / Sour / Bitter), compute properties, and
          visualize predicted structures.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.write("")  # placeholder (model metrics are shown in the Model tab)

# Tabs for clean separation of features
tabs = st.tabs([
    "Predict ✨",
    "Batch 📤",
    "Model 🔍",
    "Dataset 📂",
    "Structure 🧬",
])


# Load dataset (used for training and preview)
@st.cache_data
def load_data():
    df = pd.read_excel("AIML.xlsx")
    df.columns = df.columns.str.strip()
    return df


# make dataset available to training and preview
data = load_data()

with tabs[3]:
    st.subheader("📂 Dataset Preview")
    st.dataframe(data.head(25))


# Core computational helpers are imported from `peptide_core.py` to allow testing.
# (They are imported at top-level)

# -----------------------------
# 4. Train / Load Model
# -----------------------------
MODEL_PATH = "taste_model.pkl"



@st.cache_resource
def train_or_load_model():
    # Ensure data variable exists
    if 'data' not in globals():
        st.error("Dataset not found. Ensure `AIML.xlsx` exists in the repo root.")
        return None, None, None, None, None

    # If we already have a model file, load it to save startup time
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            X_full = compute_features(data["peptide"].tolist())
            y_full = data["Taste"]
            # get quick evaluation on full data
            try:
                y_pred_full = model.predict(X_full)
                acc_full = accuracy_score(y_full, y_pred_full)
                report_full = classification_report(y_full, y_pred_full, output_dict=True)
            except Exception:
                acc_full = None
                report_full = None
            return model, X_full, y_full, acc_full, report_full
        except Exception as e:
            st.warning(f"Existing model could not be loaded ({e}), retraining...")

    # Otherwise train a fresh model
    X = compute_features(data["peptide"].tolist())
    y = data["Taste"]

    if X.empty:
        st.error("Feature computation failed or returned empty DataFrame.")
        return None, None, None, None, None

    # Remove classes with <2 samples
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    valid_indices = y[y.isin(valid_classes)].index
    X = X.loc[valid_indices]
    y = y[valid_indices]

    if len(y.unique()) < 2:
        st.error("Not enough classes with sufficient samples to train a model.")
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, X, y, acc, report



# Load or train model
model, X, y, acc, report = train_or_load_model()

with tabs[2]:
    st.subheader("🔎 Model Metrics")
    if model:
        with st.expander("Classification report", expanded=False):
            if acc is not None:
                st.success(f"✅ Model trained with accuracy: {acc:.2f}")
            st.write("📊 Classification Report")
            st.json(report)

        with st.expander("Top 20 Feature Importances", expanded=True):
            feat_importances = pd.Series(model.feature_importances_, index=X.columns)
            feat_importances = feat_importances.sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax)
            ax.set_title("Top 20 Feature Importances")
            st.pyplot(fig)

        with st.expander("Confusion Matrix", expanded=False):
            y_pred_all = model.predict(X)
            cm = confusion_matrix(y, y_pred_all, labels=model.classes_)
            cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            ax2.set_title("Confusion Matrix")
            st.pyplot(fig2)
    else:
        st.info("Model not available. Train or ensure `AIML.xlsx` has sufficient data.")

# Allow sidebar sample button to populate the input
if st.session_state.get("sample_btn", False):
    st.session_state["user_seq"] = sample

with tabs[0]:
    st.subheader("🔬 Predict Taste and Properties of a Peptide")
    st.caption("Enter a peptide sequence (one-letter codes). Sequences are validated before prediction.")

    col_main, col_side = st.columns([2, 1])
    with col_main:
        user_seq = st.text_input("Enter peptide sequence (e.g., EDEGEQPRPF)", key="user_seq")
        if st.button("Predict", key="predict_btn"):
            if not user_seq:
                st.warning("Please enter a peptide sequence.")
            else:
                valid, msg = validate_sequence(user_seq)
                if not valid:
                    st.error(msg)
                else:
                    feats = compute_features([msg])
                    if not feats.empty and model is not None:
                        prediction = model.predict(feats)[0]
                        probs = model.predict_proba(feats)[0]

                        pred_html = (
                            "<div class='card' role='status' aria-live='polite'>"
                            f"<h3 style='margin:0'>Predicted Taste: <span class='accent'>{prediction}</span></h3>"
                            "</div>"
                        )
                        st.markdown(pred_html, unsafe_allow_html=True)

                        st.markdown("**Class probabilities (top 6)**")
                        top_idx = np.argsort(probs)[::-1][:6]
                        for i in top_idx:
                            cls = model.classes_[i]
                            p = probs[i]
                            pct = int(round(p * 100))
                            # Choose color based on score for better semantics & accessibility
                            if pct >= 60:
                                bg = "#198754"  # strong green
                            elif pct >= 30:
                                bg = "#0d6efd"  # primary blue
                            else:
                                bg = "#fd7e14"  # orange low

                            bar_html = (
                                "<div class='card small' style='margin-bottom:8px'>"
                                f"<strong>{cls}</strong> <span style='float:right'>{pct}%</span>"
                                "<div style='background:#e9ecef; border-radius:6px; height:12px; margin-top:6px;'>"
                                f"<div role='progressbar' aria-valuemin='0' aria-valuemax='100' aria-valuenow='{pct}' "
                                f"style='width:{pct}%; background:{bg}; height:100%; border-radius:6px;'></div>"
                                "</div></div>"
                            )
                            st.markdown(bar_html, unsafe_allow_html=True)

                        # Compute properties and docking score, show in the side column
                        with col_side:
                            st.subheader("🧪 Properties")
                            try:
                                props = compute_peptide_properties(msg)

                                # Docking score computed from GRAVY (hydrophobicity) and model confidence
                                from peptide_core import compute_docking_score
                                confidence = float(np.max(probs))
                                gravy = props.get("Gravy (Hydrophobicity)", 0.0)
                                docking_score = compute_docking_score(gravy, confidence)

                                props["Docking score (pred)"] = f"{docking_score} / 100"

                                st.table(pd.DataFrame.from_dict(props, orient='index', columns=['Value']))

                                # Visual indicator
                                try:
                                    st.progress(docking_score)
                                except Exception:
                                    # Some Streamlit versions expect 0.0-1.0; fall back gracefully
                                    st.progress(docking_score / 100.0)

                                st.caption("Docking score is a heuristic combining model confidence and peptide hydrophobicity.")

                            except Exception as e:
                                st.error(f"Property calculation failed: {e}")

                        # Predicted 3-letter sequence
                        st.subheader("🧬 3-letter Representation")
                        try:
                            aa3_list = [seq3(res) for res in msg]
                            st.write(" - ".join(aa3_list))
                            st.caption("A true 3D structure requires external tools like AlphaFold.")
                        except Exception as e:
                            st.error(f"Could not generate structure: {e}")
                    else:
                        st.warning("Could not compute features for the entered sequence or model unavailable.")

with tabs[1]:
    st.subheader("📤 Batch Prediction from File")
    uploader_prompt = "Upload a CSV or Excel file with a 'peptide' column"
    uploaded = st.file_uploader(uploader_prompt, key="batch_upload")

    if uploaded:
        if uploaded.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)

        df_up.columns = df_up.columns.str.strip()

        if "peptide" not in df_up.columns:
            st.error("File must contain a 'peptide' column.")
        else:
            # Normalize sequences and validate
            df_up["_peptide_norm"] = (
                df_up["peptide"].astype(str).str.strip().str.upper()
            )
            df_up["_valid"] = df_up["_peptide_norm"].apply(
                lambda s: validate_sequence(s)[0]
            )

            if not df_up["_valid"].any():
                st.error("No valid peptide sequences found (check allowed amino acid letters).")
            else:
                with st.spinner("Computing features and predicting..."):
                    feats_up = compute_features(df_up.loc[df_up["_valid"], "_peptide_norm"].tolist())
                if not feats_up.empty and model is not None:
                    preds = model.predict(feats_up)
                    df_up.loc[df_up["_valid"], "Predicted_Taste"] = preds

                    st.write("Batch Predictions (invalid sequences are marked)")
                    st.dataframe(df_up.drop(columns=["_peptide_norm"]))

                    st.download_button(
                        "Download Results",
                        df_up.drop(columns=["_peptide_norm"]).to_csv(index=False).encode(),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                    if (~df_up["_valid"]).any():
                        st.warning("Some rows contained invalid sequences and were not predicted.")
                else:
                    st.error("Could not compute features for the sequences in the uploaded file or model unavailable.")
with tabs[4]:
    st.subheader("🧬 AlphaFold/ColabFold Predicted Structure")

    st.markdown(
        "You can generate a predicted 3D structure for your peptide using "
        "**[ColabFold]("
        "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2_advanced.ipynb"
        ")**:\n"
        "1. Open the link above in Google Colab.\n"
        "2. Paste your peptide sequence in FASTA format.\n"
        "3. Run the notebook (needs a GPU runtime).\n"
        "4. Download the predicted PDB file.\n"
        "5. Upload the PDB file below to visualize it here."
    )

    uploaded_pdb = st.file_uploader("Upload ColabFold/AlphaFold PDB file (ligand)", type=["pdb"], key="pdb_upload")

    def show_structure(pdb_text):
        """Render a 3D structure from a PDB string using py3Dmol."""
        view = py3Dmol.view(width=600, height=500)
        view.addModel(pdb_text, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        return view

    # Show uploaded ligand structure viewer (if any)
    if uploaded_pdb is not None:
        pdb_content = uploaded_pdb.read().decode("utf-8")
        viewer = show_structure(pdb_content)
        st.components.v1.html(viewer._make_html(), height=400)
    else:
        st.info("No PDB uploaded yet. Use ColabFold to generate one.")
