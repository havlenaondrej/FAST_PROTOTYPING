import os
import json
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI


@st.cache_data(show_spinner=False)
def load_reviews_csv(csv_path: str) -> pd.DataFrame:
	if not Path(csv_path).exists():
		return pd.DataFrame()
	try:
		df = pd.read_csv(csv_path)
		# Drop duplicate-named columns to avoid Arrow errors in Streamlit
		if hasattr(df.columns, "duplicated"):
			df = df.loc[:, ~df.columns.duplicated()].copy()
		return df
	except Exception as exc:
		st.error(f"Failed to read CSV at {csv_path}: {exc}")
		return pd.DataFrame()


def guess_column(df: pd.DataFrame, candidates: List[str]) -> str:
	cols_lower = {c.lower(): c for c in df.columns}
	for cand in candidates:
		for lower, original in cols_lower.items():
			if cand in lower:
				return original
	return df.columns[0] if len(df.columns) else ""


def normalize_label(label: str) -> str:
	label_lower = str(label).strip().lower()
	if label_lower in {"positive", "pos", "+"}:
		return "positive"
	if label_lower in {"negative", "neg", "-"}:
		return "negative"
	return "neutral"


def classify_batch_with_openai(texts: List[str], model: str = "gpt-4o-mini") -> List[str]:
	client = OpenAI()
	labels: List[str] = []
	for txt in texts:
		try:
			resp = client.chat.completions.create(
				model=model,
				response_format={"type": "json_object"},
				messages=[
					{"role": "system", "content": (
						"You are a concise sentiment classifier. "
						"Return strict JSON: {\"label\": \"positive|neutral|negative\"}."
					)},
					{"role": "user", "content": txt[:4000]},
				],
				temperature=0,
			)
			content = resp.choices[0].message.content or "{}"
			obj = json.loads(content)
			labels.append(normalize_label(obj.get("label", "neutral")))
		except Exception:
			labels.append("neutral")
	return labels


@st.cache_data(show_spinner=False)
def cached_sentiment(texts: List[str], model: str) -> List[str]:
	return classify_batch_with_openai(texts, model=model)


def main() -> None:
	st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
	st.title("GenAI Sentiment Dashboard")
	st.caption("Load a real CSV, classify sentiment via OpenAI, filter by product, and visualize.")

	load_dotenv(override=False)
	# Support Streamlit Cloud secrets as a fallback for the API key
	try:
		if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
			os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
	except Exception:
		pass
	base_dir = Path(__file__).resolve().parent
	default_csv = str(base_dir / "data" / "customer_reviews.csv")

	# Sidebar controls
	st.sidebar.header("Controls")
	csv_path = st.sidebar.text_input("CSV path", value=default_csv, key="csv_path")
	load_btn = st.sidebar.button("1) Load dataset", type="primary", key="btn_load")
	analyze_btn = st.sidebar.button("2) Analyze sentiment (on filtered)", key="btn_analyze")
	model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini"], index=0, key="model")

	# Session storage
	if "df" not in st.session_state:
		st.session_state.df = pd.DataFrame()
	if "annotated" not in st.session_state:
		st.session_state.annotated = pd.DataFrame()

	# Step 1: Load dataset (no row limit)
	if load_btn:
		st.session_state.df = load_reviews_csv(csv_path)
		st.session_state.annotated = pd.DataFrame()

	df = st.session_state.df
	if df.empty:
		st.info("Click '1) Load dataset' to load the first 20 rows.")
		st.stop()

	# Column mapping
	product_col_guess = guess_column(df, ["product", "item", "sku"])
	text_col_guess = guess_column(df, ["review", "text", "summary", "comment", "feedback"])
	st.sidebar.subheader("Column mapping")
	product_col = st.sidebar.selectbox("Product column", options=df.columns.tolist(), index=df.columns.get_loc(product_col_guess) if product_col_guess in df.columns else 0, key="product_col")
	text_col = st.sidebar.selectbox("Text column", options=df.columns.tolist(), index=df.columns.get_loc(text_col_guess) if text_col_guess in df.columns else 0, key="text_col")

	# Product filter (UX-friendly) — applied BEFORE analysis
	products = sorted(pd.Series(df[product_col]).dropna().astype(str).unique().tolist())
	selected_products = st.multiselect(
		"Filter products",
		options=products,
		default=products,
		key="product_filter",
		help="Choose one or more products to display",
	)
	filtered_base = df[df[product_col].astype(str).isin(selected_products)] if selected_products else df

	# Step 2: Analyze sentiment (cached) on the FILTERED subset; only run when button clicked
	if analyze_btn:
		if not os.getenv("OPENAI_API_KEY"):
			st.error("OPENAI_API_KEY not set in environment/.env")
		else:
			with st.spinner("Analyzing sentiment with OpenAI on filtered data..."):
				labels = cached_sentiment(filtered_base[text_col].astype(str).tolist(), model_name)
				st.session_state.annotated = filtered_base.assign(sentiment=labels)

	# Choose what to display: if we have annotated results, respect current filter; else show filtered base
	if not st.session_state.annotated.empty:
		annotated = st.session_state.annotated
		# Re-apply current filter to annotated (in case user changed filter after running)
		filtered = annotated[annotated[product_col].astype(str).isin(selected_products)] if selected_products else annotated
	else:
		filtered = filtered_base

	# Layout: chart on left, table on right
	left, right = st.columns([1, 1])
	with left:
		st.subheader("Sentiment distribution")
		if "sentiment" in filtered.columns:
			order = pd.CategoricalDtype(categories=["positive", "neutral", "negative"], ordered=True)
			counts = (
				filtered.assign(sentiment=filtered["sentiment"].astype(order))
				.groupby("sentiment", observed=False)
				.size()
				.reset_index(name="count")
			)
			palette = {
				"positive": "#16a34a",  # green
				"neutral": "#64748b",   # slate
				"negative": "#dc2626",  # red
			}
			chart = (
				alt.Chart(counts)
				.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
				.encode(
					x=alt.X("sentiment:N", sort=["positive", "neutral", "negative"], title="Sentiment"),
					y=alt.Y("count:Q", title="Reviews"),
					color=alt.Color("sentiment:N", scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values())), legend=None),
					tooltip=["sentiment", "count"],
				)
				.properties(height=280)
			)
			st.altair_chart(chart, use_container_width=True)
		else:
			st.info("No sentiment to visualize yet.")

	with right:
		st.subheader("Data preview")
		preview_cols = [c for c in [product_col, text_col, "sentiment"] if c in filtered.columns]
		# Drop duplicate-named columns for safe rendering
		table_df = filtered.loc[:, ~filtered.columns.duplicated()]
		st.dataframe(table_df[preview_cols] if preview_cols else table_df.head(20), use_container_width=True, height=320)


if __name__ == "__main__":
	main()

