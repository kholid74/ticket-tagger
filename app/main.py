"""
Auto Tagging Ticket Support System — Streamlit Dashboard

Tabs:
  1. Home           — Overview, dataset stats, quick demo
  2. Auto-Tag Demo  — Predict tag for a ticket + confidence
  3. EDA Explorer   — Distribution, word cloud, top keywords
  4. Model Performance — Confusion matrix, classification report
"""

import sys
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ─── Path setup ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Tagging Ticket Support",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .tag-badge {
        display: inline-block;
        background: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .confidence-text {
        font-size: 1.4rem;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Loaders (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = ROOT / "models" / "model.pkl"
    if not model_path.exists():
        return None
    import joblib
    return joblib.load(model_path)


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    processed = ROOT / "data" / "processed" / "tickets_clean.csv"
    raw = ROOT / "data" / "raw" / "customer_support_tickets.csv"

    if processed.exists():
        return pd.read_csv(processed)
    elif raw.exists():
        from src.preprocessing import load_and_prepare_data, preprocess_series
        df = load_and_prepare_data(str(raw))
        df["text_clean"] = preprocess_series(df["text"])
        return df
    return None


@st.cache_data(show_spinner="Loading metrics...")
def load_metrics():
    path = ROOT / "models" / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def predict(text: str):
    from src.preprocessing import preprocess_text
    pipeline = load_model()
    if pipeline is None:
        return None
    processed = preprocess_text(text)
    tag = pipeline.predict([processed])[0]
    proba = pipeline.predict_proba([processed])[0]
    classes = pipeline.classes_
    all_scores = {cls: float(prob) for cls, prob in zip(classes, proba)}
    confidence = float(np.max(proba))
    return {"tag": tag, "confidence": confidence, "all_scores": all_scores}


# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/technical-support.png", width=80)
    st.title("Ticket Tagger")
    st.caption("Auto Tagging Ticket Support System")
    st.divider()

    tab_selected = st.radio(
        "Navigation",
        ["🏠 Home", "🎯 Auto-Tag Demo", "🤖 AI Reply Suggester", "📊 EDA Explorer", "📈 Model Performance"],
        label_visibility="collapsed",
    )

    st.divider()
    model_loaded = load_model() is not None
    data_loaded = load_data() is not None
    metrics_loaded = load_metrics() is not None

    st.markdown("**System Status**")
    st.markdown(f"{'✅' if model_loaded else '❌'} Model {'loaded' if model_loaded else 'not found'}")
    st.markdown(f"{'✅' if data_loaded else '⚠️'} Dataset {'available' if data_loaded else 'not found'}")
    st.markdown(f"{'✅' if metrics_loaded else '⚠️'} Metrics {'available' if metrics_loaded else 'not found'}")
    gemini_configured = bool(os.environ.get("GEMINI_API_KEY", ""))
    st.markdown(f"{'✅' if gemini_configured else '⚠️'} Gemini API {'configured' if gemini_configured else 'not set'}")

    if not model_loaded:
        st.warning("Run `python src/train.py` to train the model first.")


# ══════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════
if tab_selected == "🏠 Home":
    st.title("🎫 Auto Tagging Ticket Support System")
    st.markdown(
        """
        An intelligent system that **automatically classifies** customer support tickets
        into the correct category using Machine Learning — saving support agents time and
        ensuring tickets are routed to the right team immediately.
        """
    )

    st.divider()

    # ── System Overview ──
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("How It Works")
        st.markdown(
            """
            ```
            Customer submits ticket text
                    ↓
            Text Preprocessing
            (lowercase → remove noise → tokenize → lemmatize)
                    ↓
            TF-IDF Vectorization
            (convert text to numerical features)
                    ↓
            Logistic Regression Classifier
                    ↓
            Predicted Tag + Confidence Score
            ```
            """
        )

    with col2:
        st.subheader("Tech Stack")
        st.markdown(
            """
            - **Model**: TF-IDF + Logistic Regression
            - **NLP**: NLTK (stopwords, lemmatization)
            - **UI**: Streamlit
            - **Viz**: Plotly, WordCloud
            - **Deploy**: GCP Cloud Run
            """
        )

    st.divider()

    # ── Dataset Stats ──
    df = load_data()
    metrics = load_metrics()

    if df is not None:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets", f"{len(df):,}")
        col2.metric("Unique Categories", df["label"].nunique())
        col3.metric("Avg Text Length", f"{df['text'].str.len().mean():.0f} chars")
        col4.metric(
            "Model F1-score",
            f"{metrics['f1_weighted']:.2%}" if metrics else "N/A",
            delta="↑ weighted" if metrics else None,
        )

        st.divider()

        # Category distribution (quick preview)
        st.subheader("Category Distribution")
        cat_counts = df["label"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(
            cat_counts,
            x="Count",
            y="Category",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            title="Ticket Count per Category",
        )
        fig.update_layout(showlegend=False, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Dataset not found. Place `customer_support_tickets.csv` in `data/raw/` to see stats.")

    # ── Quick Demo ──
    st.divider()
    st.subheader("Quick Demo")
    demo_text = st.text_area(
        "Enter a support ticket:",
        value="My internet connection keeps dropping every few minutes and I can't work from home.",
        height=80,
    )
    if st.button("Predict Tag", type="primary"):
        if load_model() is None:
            st.error("Model not loaded. Run `python src/train.py` first.")
        else:
            result = predict(demo_text)
            if result:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f'<div class="tag-badge">🏷️ {result["tag"]}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="confidence-text">{result["confidence"]:.1%} confident</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    scores = dict(sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True)[:5])
                    fig = px.bar(
                        x=list(scores.values()),
                        y=list(scores.keys()),
                        orientation="h",
                        labels={"x": "Confidence", "y": "Tag"},
                        color=list(scores.values()),
                        color_continuous_scale="Greens",
                    )
                    fig.update_layout(height=250, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Go to **Auto-Tag Demo** tab for detailed predictions.")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — AUTO-TAG DEMO
# ══════════════════════════════════════════════════════════════════
elif tab_selected == "🎯 Auto-Tag Demo":
    st.title("🎯 Auto-Tag Demo")
    st.markdown("Enter a support ticket text to see the predicted tag and confidence scores.")

    if load_model() is None:
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    # ── Sample tickets ──
    sample_options = {
        "Technical Issue": "My laptop screen keeps flickering and sometimes goes completely black during video calls.",
        "Billing": "I was charged twice for my subscription this month. Please reverse the duplicate charge.",
        "Account": "I cannot log into my account. The password reset email is not arriving in my inbox.",
        "Shipping": "My order was supposed to be delivered 3 days ago but the tracking shows it's still in transit.",
        "Cancellation": "I want to cancel my subscription effective immediately and get a prorated refund.",
        "Custom": "",
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        sample_choice = st.selectbox("Load a sample ticket:", list(sample_options.keys()))

    if sample_choice != "Custom":
        ticket_text = sample_options[sample_choice]
    else:
        ticket_text = ""

    ticket_input = st.text_area(
        "Support Ticket Text:",
        value=ticket_text,
        height=120,
        placeholder="Type or paste the customer support ticket here...",
    )

    predict_btn = st.button("🔍 Predict Tag", type="primary", use_container_width=True)

    if predict_btn and ticket_input.strip():
        with st.spinner("Analyzing ticket..."):
            result = predict(ticket_input)

        if result:
            st.divider()
            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.metric("Predicted Tag", result["tag"])

            with col2:
                confidence_pct = result["confidence"] * 100
                if confidence_pct >= 80:
                    color = "🟢"
                elif confidence_pct >= 60:
                    color = "🟡"
                else:
                    color = "🔴"
                st.metric("Confidence", f"{color} {confidence_pct:.1f}%")

            with col3:
                # Confidence gauge
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence_pct,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#4CAF50"},
                            "steps": [
                                {"range": [0, 60], "color": "#ffcccc"},
                                {"range": [60, 80], "color": "#fff3cc"},
                                {"range": [80, 100], "color": "#ccffcc"},
                            ],
                            "threshold": {
                                "line": {"color": "green", "width": 4},
                                "thickness": 0.75,
                                "value": 80,
                            },
                        },
                        title={"text": "Confidence Level"},
                    )
                )
                fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

            # All scores bar chart
            st.subheader("All Category Scores")
            all_scores = dict(sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True))
            fig2 = px.bar(
                x=list(all_scores.values()),
                y=list(all_scores.keys()),
                orientation="h",
                labels={"x": "Confidence Score", "y": "Category"},
                color=list(all_scores.values()),
                color_continuous_scale="RdYlGn",
                range_color=[0, 1],
            )
            fig2.update_layout(height=max(300, len(all_scores) * 40), coloraxis_showscale=True)
            st.plotly_chart(fig2, use_container_width=True)

    elif predict_btn:
        st.warning("Please enter a ticket text before predicting.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — AI REPLY SUGGESTER
# ══════════════════════════════════════════════════════════════════
elif tab_selected == "🤖 AI Reply Suggester":
    st.title("🤖 AI Reply Suggester")
    st.markdown(
        "Paste a customer support ticket — the system will **auto-tag** it, "
        "then use **Gemini AI** to draft a professional reply for the support agent."
    )

    if load_model() is None:
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    # ── Gemini setup ──────────────────────────────────────────────
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        st.warning(
            "GEMINI_API_KEY not set. "
            "Get your key at https://aistudio.google.com/app/apikey, "
            "then set it as an environment variable or in Cloud Run."
        )
        st.stop()

    @st.cache_resource(show_spinner=False)
    def get_gemini_client():
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        return genai.GenerativeModel("gemini-2.5-flash")

    def generate_reply(ticket_text: str, tag: str, confidence: float) -> str:
        model_gemini = get_gemini_client()
        prompt = f"""You are a professional and empathetic customer support agent.

A customer submitted the following support ticket, which has been automatically classified as: **{tag}** (confidence: {confidence:.0%}).

Customer Ticket:
\"\"\"
{ticket_text}
\"\"\"

Write a concise, professional, and empathetic reply to this customer. The reply should:
1. Acknowledge the customer's issue
2. Provide a helpful response or next steps based on the '{tag}' category
3. Be polite and solution-oriented
4. Be between 3-5 sentences

Reply only with the email body — no subject line, no greeting like "Dear Customer", start directly with the response."""

        response = model_gemini.generate_content(prompt)
        return response.text.strip()

    # ── Sample tickets ─────────────────────────────────────────────
    sample_options = {
        "Technical Issue": "My laptop screen keeps flickering and sometimes goes completely black during video calls.",
        "Billing": "I was charged twice for my subscription this month. Please reverse the duplicate charge.",
        "Account": "I cannot log into my account. The password reset email is not arriving in my inbox.",
        "Shipping": "My order was supposed to be delivered 3 days ago but the tracking shows it's still in transit.",
        "Cancellation": "I want to cancel my subscription effective immediately and get a prorated refund.",
        "Custom": "",
    }

    col1, _ = st.columns([1, 2])
    with col1:
        sample_choice = st.selectbox("Load a sample ticket:", list(sample_options.keys()), key="ai_sample")

    ticket_text = sample_options[sample_choice] if sample_choice != "Custom" else ""

    ticket_input = st.text_area(
        "Customer Ticket:",
        value=ticket_text,
        height=130,
        placeholder="Paste the customer support ticket here...",
    )

    generate_btn = st.button("✨ Auto-Tag & Generate Reply", type="primary", use_container_width=True)

    if generate_btn and ticket_input.strip():
        with st.spinner("Analyzing ticket and generating reply..."):
            result = predict(ticket_input)

            if result:
                col1, col2 = st.columns(2)
                with col1:
                    confidence_pct = result["confidence"] * 100
                    color = "🟢" if confidence_pct >= 80 else ("🟡" if confidence_pct >= 60 else "🔴")
                    st.metric("Auto-Tag", result["tag"])
                    st.metric("Confidence", f"{color} {confidence_pct:.1f}%")

                with col2:
                    scores = dict(sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True))
                    fig = px.bar(
                        x=list(scores.values()),
                        y=list(scores.keys()),
                        orientation="h",
                        labels={"x": "Score", "y": "Category"},
                        color=list(scores.values()),
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 1],
                    )
                    fig.update_layout(height=220, coloraxis_showscale=False, margin=dict(t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("Suggested Reply")

                try:
                    reply = generate_reply(ticket_input, result["tag"], result["confidence"])
                    st.text_area("Draft Reply (editable):", value=reply, height=180, key="reply_output")
                    st.caption("This reply was generated by Gemini AI. Always review before sending to the customer.")
                except Exception as e:
                    st.error(f"Failed to generate reply: {e}")

    elif generate_btn:
        st.warning("Please enter a ticket text first.")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════
elif tab_selected == "📊 EDA Explorer":
    st.title("📊 EDA Explorer")
    st.markdown("Explore the distribution and characteristics of the training dataset.")

    df = load_data()
    if df is None:
        st.error("Dataset not found. Place `customer_support_tickets.csv` in `data/raw/`.")
        st.stop()

    # ── Class Distribution ──
    st.subheader("1. Class Distribution")
    cat_counts = df["label"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_counts["Percentage"] = (cat_counts["Count"] / len(df) * 100).round(2)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            cat_counts,
            x="Category",
            y="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="Ticket Count per Category",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            title="Category Share (%)",
            hole=0.4,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(cat_counts, use_container_width=True, hide_index=True)

    # ── Text Length Analysis ──
    st.subheader("2. Text Length Analysis")
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df,
            x="text_length",
            nbins=50,
            title="Distribution of Text Length (chars)",
            color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.box(
            df,
            x="label",
            y="word_count",
            title="Word Count per Category",
            color="label",
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Word Cloud ──
    st.subheader("3. Word Cloud")
    selected_cat = st.selectbox("Select Category:", ["All"] + sorted(df["label"].unique().tolist()))

    text_col = "text_clean" if "text_clean" in df.columns else "text"

    if selected_cat == "All":
        corpus = " ".join(df[text_col].fillna("").tolist())
    else:
        corpus = " ".join(df[df["label"] == selected_cat][text_col].fillna("").tolist())

    if corpus.strip():
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=150,
        ).generate(corpus)

        fig_wc, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {selected_cat}", fontsize=14)
        st.pyplot(fig_wc)
        plt.close()

    # ── Top Keywords per Category ──
    st.subheader("4. Top Keywords per Category")
    from collections import Counter

    cat_for_keywords = st.selectbox("Select Category for Keywords:", sorted(df["label"].unique().tolist()))
    cat_text = " ".join(df[df["label"] == cat_for_keywords][text_col].fillna("").tolist())
    word_freq = Counter(cat_text.split()).most_common(20)

    if word_freq:
        kw_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
        fig = px.bar(
            kw_df,
            x="Frequency",
            y="Word",
            orientation="h",
            title=f"Top 20 Keywords — {cat_for_keywords}",
            color="Frequency",
            color_continuous_scale="Purples",
        )
        fig.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── Sample Data ──
    st.subheader("5. Sample Data")
    sample_cat = st.selectbox("Show samples from:", ["All"] + sorted(df["label"].unique().tolist()), key="sample_cat")
    n_samples = st.slider("Number of samples:", 3, 20, 5)

    if sample_cat == "All":
        sample_df = df.sample(min(n_samples, len(df)), random_state=42)
    else:
        filtered = df[df["label"] == sample_cat]
        sample_df = filtered.sample(min(n_samples, len(filtered)), random_state=42)

    st.dataframe(
        sample_df[["label", "text"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
elif tab_selected == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Evaluation metrics from the test set (20% holdout).")

    metrics = load_metrics()
    if metrics is None:
        st.error("Metrics not found. Run `python src/train.py` to generate metrics.")
        st.stop()

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("F1-score (weighted)", f"{metrics['f1_weighted']:.2%}")
    col3.metric("Train Samples", f"{metrics['train_size']:,}")
    col4.metric("Test Samples", f"{metrics['test_size']:,}")

    st.divider()

    # ── Per-class metrics ──
    st.subheader("1. Per-Class F1 Scores")
    report = metrics["classification_report"]
    class_metrics = []
    for label, vals in report.items():
        if isinstance(vals, dict) and "f1-score" in vals:
            class_metrics.append(
                {
                    "Category": label,
                    "Precision": round(vals["precision"], 3),
                    "Recall": round(vals["recall"], 3),
                    "F1-Score": round(vals["f1-score"], 3),
                    "Support": int(vals["support"]),
                }
            )

    class_df = pd.DataFrame(class_metrics)
    # Filter to only actual classes (not macro/weighted averages)
    labels = metrics.get("labels", [])
    if labels:
        class_df = class_df[class_df["Category"].isin(labels)]

    fig = px.bar(
        class_df,
        x="F1-Score",
        y="Category",
        orientation="h",
        color="F1-Score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        title="F1-Score per Category",
    )
    fig.add_vline(x=0.75, line_dash="dash", line_color="red", annotation_text="Target: 0.75")
    fig.update_layout(height=max(300, len(class_df) * 35))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(class_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Confusion Matrix ──
    st.divider()
    st.subheader("2. Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]

    # Normalize
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig_cm = px.imshow(
        cm_normalized,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix (Normalized)",
        labels={"x": "Predicted", "y": "Actual", "color": "Rate"},
        text_auto=".2f",
    )
    fig_cm.update_layout(height=max(400, len(labels) * 45))
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Model info ──
    st.divider()
    st.subheader("3. Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            | Parameter | Value |
            |-----------|-------|
            | Model Type | {metrics.get('model_type', 'TF-IDF + Logistic Regression')} |
            | Number of Classes | {metrics.get('num_classes', len(labels))} |
            | Train / Test Split | 80% / 20% |
            | TF-IDF Max Features | 10,000 |
            | N-gram Range | (1, 2) |
            | Solver | lbfgs |
            """
        )
    with col2:
        # Radar chart for precision/recall/f1 per class
        if not class_df.empty:
            fig_radar = go.Figure()
            for _, row in class_df.iterrows():
                fig_radar.add_trace(
                    go.Bar(
                        name=row["Category"],
                        x=["Precision", "Recall", "F1-Score"],
                        y=[row["Precision"], row["Recall"], row["F1-Score"]],
                    )
                )
            fig_radar.update_layout(
                barmode="group",
                title="Precision / Recall / F1 per Category",
                yaxis_range=[0, 1],
                height=350,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
