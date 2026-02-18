"""
app.py  –  Raman CNN Dashboard  (Streamlit)
============================================
Run:  streamlit run app.py

Tabs
----
1. Training Monitor   – live loss/accuracy curves from training_log.csv
2. Model Info         – architecture, class distribution
3. Evaluation         – confusion matrix, classification report
4. Saliency Explorer  – per-class Integrated-Gradient maps
5. Live Inference     – classify a new spectrum from the database
"""

import os, json, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Raman CNN Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT      = "outputs"
LOG_CSV  = os.path.join(OUT, "logs",  "training_log.csv")
REG_CSV  = os.path.join(OUT, "logs",  "key_spectral_regions.csv")
SAL_NPZ  = os.path.join(OUT, "logs",  "saliency_maps.npz")
CFG_JSON = os.path.join(OUT, "model", "model_config.json")
BEST_PT  = os.path.join(OUT, "model", "best_model.pt")
CM_PNG   = os.path.join(OUT, "figures", "confusion_matrix.png")
TC_PNG   = os.path.join(OUT, "figures", "training_curves.png")
CD_PNG   = os.path.join(OUT, "figures", "class_distribution.png")

META_CSV    = "ramanbiolib/db/metadata_db.csv"
SPECTRA_CSV = "ramanbiolib/db/raman_spectra_db.csv"

# ── Helper: parse list strings from CSV ───────────────────────────────────────
def parse_list(s):
    return [float(v) for v in str(s).strip("[]").split(", ") if v]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Raman CNN Dashboard")
st.sidebar.markdown("**ramanbiolib** — 1D CNN + Integrated Gradients")
st.sidebar.divider()

# Training status badge
training_done = os.path.exists(LOG_CSV)
if training_done:
    cfg = json.load(open(CFG_JSON)) if os.path.exists(CFG_JSON) else {}
    test_acc = cfg.get('test_acc', None)
    acc_str = f"{test_acc:.1%}" if isinstance(test_acc, (int, float)) else "N/A"
    st.sidebar.success(f"✅ Training complete\nTest acc: **{acc_str}**")
else:
    st.sidebar.warning("⏳ Training in progress…")
    st.sidebar.caption("Dashboard auto-refreshes every 10 s")

if os.path.exists(BEST_PT):
    size_mb = os.path.getsize(BEST_PT) / 1e6
    st.sidebar.info(f"💾 best_model.pt  ({size_mb:.1f} MB)")

st.sidebar.divider()
tab_names = ["📈 Training Monitor", "🏗️ Model Info",
             "📊 Evaluation", "🌡️ Saliency Explorer", "⚡ Live Inference"]
selected = st.sidebar.radio("Navigate", tab_names)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Monitor
# ═══════════════════════════════════════════════════════════════════════════════
if selected == tab_names[0]:
    st.title("📈 Training Monitor")

    if not os.path.exists(LOG_CSV):
        # Show partial progress from best_model.pt existence
        st.info("Training is running in the background. This page auto-refreshes every 10 seconds.")
        col1, col2 = st.columns(2)
        col1.metric("Model checkpoint", "✅ saved" if os.path.exists(BEST_PT) else "⏳ waiting")
        col2.metric("Class distribution", "✅ saved" if os.path.exists(CD_PNG) else "⏳ waiting")
        if os.path.exists(CD_PNG):
            st.image(CD_PNG, caption="Class Distribution", use_container_width=True)
        st.info("Training log will appear here once training completes.")
        time.sleep(10)
        st.rerun()
    else:
        df = pd.read_csv(LOG_CSV)
        epochs = list(range(1, len(df) + 1))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Cross-Entropy Loss", "Classification Accuracy"])
        fig.add_trace(go.Scatter(x=epochs, y=df["train_loss"], name="Train loss", line=dict(color="#4C8BF5")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=df["val_loss"],   name="Val loss",   line=dict(color="#F5A623")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=df["train_acc"],  name="Train acc",  line=dict(color="#4C8BF5")), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=df["val_acc"],    name="Val acc",    line=dict(color="#F5A623")), row=1, col=2)
        fig.update_xaxes(title_text="Epoch")
        fig.update_layout(height=400, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final train loss", f"{df['train_loss'].iloc[-1]:.4f}")
        col2.metric("Final val loss",   f"{df['val_loss'].iloc[-1]:.4f}")
        col3.metric("Best train acc",   f"{df['train_acc'].max():.3f}")
        col4.metric("Best val acc",     f"{df['val_acc'].max():.3f}")

        st.dataframe(df.round(4), use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Info
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[1]:
    st.title("🏗️ Model & Data Info")

    # Class distribution
    if os.path.exists(CD_PNG):
        st.subheader("Class Distribution")
        st.image(CD_PNG, use_container_width=True)

    # Dataset stats
    try:
        meta = pd.read_csv(META_CSV)
        spec = pd.read_csv(SPECTRA_CSV,
                           converters={"wavenumbers": parse_list, "intensity": parse_list})
        merged = spec.merge(meta[["id","type"]].drop_duplicates("id"), on="id")
        merged["class"] = merged["type"].str.split("/").str[0]
        KEEP = ["Proteins","Lipids","Saccharides","AminoAcids","PrimaryMetabolites","NucleicAcids"]
        merged = merged[merged["class"].isin(KEEP)]
        counts = merged["class"].value_counts().reset_index()
        counts.columns = ["Class","Spectra"]

        fig = px.bar(counts, x="Class", y="Spectra", color="Class",
                     title="Spectra per class (ramanbiolib)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        wn = np.array(merged["wavenumbers"].iloc[0])
        col1, col2, col3 = st.columns(3)
        col1.metric("Total spectra (6 classes)", len(merged))
        col2.metric("Wavenumber points", len(wn))
        col3.metric("Wavenumber range", f"{wn[0]:.0f}–{wn[-1]:.0f} cm⁻¹")
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")

    # Architecture
    st.subheader("CNN Architecture")
    st.code("""
RamanCNN1D  (input: batch × 1 × 1351)
│
├─ Block 1: Conv1D(1→32, k=15) ×2  + BN + ReLU + MaxPool(4) + Dropout(0.25)
├─ Block 2: Conv1D(32→64, k=11) ×2 + BN + ReLU + MaxPool(4) + Dropout(0.25)
├─ Block 3: Conv1D(64→128, k=7)    + BN + ReLU + MaxPool(4) + Dropout(0.25)
│
├─ Flatten → Linear(→256) + ReLU + Dropout(0.4)
└─ Linear(256 → 6 classes)

Parameters : 831,654
Optimizer  : Adam  (lr=1e-3, wd=1e-4)
Scheduler  : CosineAnnealingLR (T_max=80)
Batch size : 32  (WeightedRandomSampler for class balance)
Augment    : ×15 Gaussian noise (σ=0.015) + amplitude scaling (0.85–1.15)
""", language="text")

    if os.path.exists(CFG_JSON):
        cfg = json.load(open(CFG_JSON))
        st.json(cfg)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[2]:
    st.title("📊 Evaluation")

    if not training_done:
        st.info("⏳ Waiting for training to complete…")
        time.sleep(10); st.rerun()
    else:
        if os.path.exists(CM_PNG):
            st.subheader("Confusion Matrix")
            st.image(CM_PNG, use_container_width=True)

        if os.path.exists(TC_PNG):
            st.subheader("Training Curves")
            st.image(TC_PNG, use_container_width=True)

        if os.path.exists(REG_CSV):
            st.subheader("Top Spectral Regions per Class (from Integrated Gradients)")
            df_reg = pd.read_csv(REG_CSV)
            st.dataframe(df_reg, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Saliency Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[3]:
    st.title("🌡️ Saliency Explorer (Integrated Gradients)")

    if not os.path.exists(SAL_NPZ):
        st.info("⏳ Saliency maps not yet generated. Waiting for training to complete…")
        time.sleep(10); st.rerun()
    else:
        data = np.load(SAL_NPZ, allow_pickle=True)
        wn   = data["wavenumbers"]
        classes = list(data["class_names"])

        sel_class = st.selectbox("Select molecular class", classes)
        sal = data[sel_class]
        sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

        # Load mean spectrum
        try:
            spec_df = pd.read_csv(SPECTRA_CSV,
                                  converters={"wavenumbers": parse_list, "intensity": parse_list})
            meta_df = pd.read_csv(META_CSV)
            merged  = spec_df.merge(meta_df[["id","type"]].drop_duplicates("id"), on="id")
            merged["class"] = merged["type"].str.split("/").str[0]
            X_cls = np.array(merged[merged["class"]==sel_class]["intensity"].tolist())
            mean_spec = X_cls.mean(axis=0) if len(X_cls) > 0 else np.zeros_like(wn)
        except Exception:
            mean_spec = np.zeros_like(wn)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=wn, y=mean_spec, name="Mean spectrum",
                                 line=dict(color="#4C8BF5", width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=wn, y=sal_n, name="|IG| saliency",
                                 fill="tozeroy", fillcolor="rgba(220,20,60,0.25)",
                                 line=dict(color="crimson", width=1)), secondary_y=True)
        fig.update_xaxes(title_text="Wavenumber (cm⁻¹)")
        fig.update_yaxes(title_text="Intensity", secondary_y=False)
        fig.update_yaxes(title_text="Normalised |IG|", secondary_y=True)
        fig.update_layout(title=f"Saliency Map — {sel_class}", height=450,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap of all classes
        st.subheader("All-class saliency heatmap")
        sal_matrix = np.array([(data[c]-data[c].min())/(data[c].max()-data[c].min()+1e-9) for c in classes])
        step = 5
        fig2 = px.imshow(sal_matrix[:, ::step],
                         x=[f"{v:.0f}" for v in wn[::step]],
                         y=classes,
                         color_continuous_scale="hot",
                         aspect="auto",
                         title="Integrated-Gradient Saliency (all classes)",
                         labels={"x": "Wavenumber (cm⁻¹)", "color": "|IG|"})
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        if os.path.exists(REG_CSV):
            df_reg = pd.read_csv(REG_CSV)
            st.subheader(f"Top wavenumber regions — {sel_class}")
            st.dataframe(df_reg[df_reg["class"]==sel_class].reset_index(drop=True),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Inference
# ═══════════════════════════════════════════════════════════════════════════════
elif selected == tab_names[4]:
    st.title("⚡ Live Inference")

    if not os.path.exists(BEST_PT):
        st.info("⏳ Model checkpoint not yet available.")
        time.sleep(5); st.rerun()

    try:
        import torch, torch.nn as nn

        class RamanCNN1D(nn.Module):
            def __init__(self, input_len=1351, n_classes=6):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Conv1d(1,32,15,padding=7),nn.BatchNorm1d(32),nn.ReLU(),
                    nn.Conv1d(32,32,15,padding=7),nn.BatchNorm1d(32),nn.ReLU(),
                    nn.MaxPool1d(4),nn.Dropout(0.25))
                self.block2 = nn.Sequential(
                    nn.Conv1d(32,64,11,padding=5),nn.BatchNorm1d(64),nn.ReLU(),
                    nn.Conv1d(64,64,11,padding=5),nn.BatchNorm1d(64),nn.ReLU(),
                    nn.MaxPool1d(4),nn.Dropout(0.25))
                self.block3 = nn.Sequential(
                    nn.Conv1d(64,128,7,padding=3),nn.BatchNorm1d(128),nn.ReLU(),
                    nn.MaxPool1d(4),nn.Dropout(0.25))
                dummy = torch.zeros(1,1,input_len)
                flat = self._fwd(dummy).shape[1]
                self.classifier = nn.Sequential(
                    nn.Linear(flat,256),nn.ReLU(),nn.Dropout(0.4),
                    nn.Linear(256,n_classes))
            def _fwd(self,x):
                return self.block3(self.block2(self.block1(x))).view(x.size(0),-1)
            def forward(self,x):
                return self.classifier(self._fwd(x))

        @st.cache_resource
        def load_model():
            cfg = json.load(open(CFG_JSON)) if os.path.exists(CFG_JSON) else {"input_len":1351,"n_classes":6,"class_names":["AminoAcids","Lipids","NucleicAcids","PrimaryMetabolites","Proteins","Saccharides"]}
            m = RamanCNN1D(cfg["input_len"], cfg["n_classes"])
            m.load_state_dict(torch.load(BEST_PT, map_location="cpu"))
            m.eval()
            return m, cfg["class_names"]

        model, CLASS_NAMES = load_model()

        # Load DB spectra
        spec_df = pd.read_csv(SPECTRA_CSV,
                              converters={"wavenumbers":parse_list,"intensity":parse_list})
        meta_df = pd.read_csv(META_CSV)
        merged = spec_df.merge(meta_df[["id","type"]].drop_duplicates("id"), on="id")
        merged["class"] = merged["type"].str.split("/").str[0]
        KEEP = ["Proteins","Lipids","Saccharides","AminoAcids","PrimaryMetabolites","NucleicAcids"]
        merged = merged[merged["class"].isin(KEEP)]

        st.subheader("Select a spectrum from the database")
        component_list = merged["component"].tolist()
        selected_comp = st.selectbox("Component", component_list)
        row = merged[merged["component"] == selected_comp].iloc[0]
        intensity = np.array(row["intensity"], dtype=np.float32)
        wn = np.array(row["wavenumbers"])
        true_class = row["class"]

        # Plot spectrum
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wn, y=intensity, mode="lines", name=selected_comp,
                                 line=dict(color="#4C8BF5", width=1.5)))
        fig.update_layout(title=f"Spectrum: {selected_comp}",
                          xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity",
                          height=350)
        st.plotly_chart(fig, use_container_width=True)

        if st.button("🔍 Run Inference", type="primary"):
            x = torch.tensor(intensity).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1).squeeze().numpy()
                pred_idx = int(probs.argmax())
                pred_class = CLASS_NAMES[pred_idx]

            col1, col2 = st.columns(2)
            match = pred_class == true_class
            col1.metric("Predicted class", pred_class,
                        delta="✓ correct" if match else f"✗ true: {true_class}",
                        delta_color="normal" if match else "inverse")
            col2.metric("Confidence", f"{probs[pred_idx]*100:.1f}%")

            fig_prob = px.bar(
                x=CLASS_NAMES, y=probs*100,
                labels={"x":"Class","y":"Probability (%)"},
                title="Class probabilities",
                color=CLASS_NAMES,
                height=300
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    except Exception as e:
        st.error(f"Inference error: {e}")
        st.info("Ensure the model checkpoint (best_model.pt) exists and training has completed.")
