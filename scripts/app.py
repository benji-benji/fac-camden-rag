import streamlit as st
import requests
import sys
from pathlib import Path
from statistics import mean
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.metrics import load_golden_dataset, evaluate_query_ranking


def init_state():
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"


def render_sidebar_stats():
    st.sidebar.subheader("Corpus status")
    try:
        response = requests.get(f"{st.session_state.api_base_url}/stats", timeout=10)
        response.raise_for_status()
        stats = response.json()
        st.sidebar.metric("Documents", stats.get("document_count", "?"))
        st.sidebar.metric("Chunks", stats.get("chunk_count", "?"))
        st.sidebar.metric("Avg tokens/chunk", stats.get("avg_chunk_tokens", "?"))
        last = stats.get("last_ingested_at", "")
        if last:
            st.sidebar.caption(f"Last ingested: {last[:19].replace('T', ' ')}")
    except Exception as e:
        st.sidebar.warning(f"Stats unavailable: {e}")


def render_batch_eval(golden_df, relevant_threshold):
    st.markdown("---")
    st.subheader("Batch eval")
    st.caption(
        "Runs every unique query in the golden dataset through the full RAG pipeline "
        "and aggregates the metrics. This is slow — each query takes ~5-15 seconds."
    )

    queries = sorted(golden_df["query_text"].unique().tolist())
    st.caption(f"Dataset has {len(queries)} unique queries.")

    max_to_run = st.slider(
        "How many queries to run",
        min_value=1, max_value=len(queries), value=min(len(queries), 10),
        help="Start with a small number — each query calls the generator so it takes time.",
    )

    if st.button("Run batch eval"):
        api = st.session_state.api_base_url
        progress = st.progress(0.0, text="Starting...")
        per_query_rows = []
        metric_series = {"map": [], "mar": [], "ndcg@5": [], "p@5": [], "r@5": [], "f1@5": []}

        for i, query in enumerate(queries[:max_to_run]):
            progress.progress((i + 1) / max_to_run, text=f"[{i+1}/{max_to_run}] {query[:60]}")
            try:
                qr = requests.post(f"{api}/query", json={"query": query, "top_k": 5}, timeout=120)
                qr.raise_for_status()
                query_id = qr.json()["query_id"]
                log = requests.get(f"{api}/logs/{query_id}", timeout=60).json()
                reranked = log.get("filters", {}).get("reranked_candidates", [])
                predicted_ids = [r["chunk_id"] for r in reranked]

                res = evaluate_query_ranking(
                    query_text=query,
                    predicted_chunk_ids=predicted_ids,
                    golden_df=golden_df,
                    ks=[1, 3, 5, 10],
                    relevant_threshold=float(relevant_threshold),
                )
                per_query_rows.append({
                    "query": query[:60],
                    "MAP": round(res.average_precision, 3),
                    "MAR": round(res.mar, 3),
                    "NDCG@5": round(res.ndcg_at_k.get(5, 0.0), 3),
                    "P@5": round(res.precision_at_k.get(5, 0.0), 3),
                    "R@5": round(res.recall_at_k.get(5, 0.0), 3),
                    "F1@5": round(res.f1_at_k.get(5, 0.0), 3),
                })
                metric_series["map"].append(res.average_precision)
                metric_series["mar"].append(res.mar)
                metric_series["ndcg@5"].append(res.ndcg_at_k.get(5, 0.0))
                metric_series["p@5"].append(res.precision_at_k.get(5, 0.0))
                metric_series["r@5"].append(res.recall_at_k.get(5, 0.0))
                metric_series["f1@5"].append(res.f1_at_k.get(5, 0.0))
            except Exception as e:
                per_query_rows.append({"query": query[:60], "error": str(e)[:80]})

        progress.empty()

        st.markdown("### Aggregate metrics")
        c1, c2, c3 = st.columns(3)
        if metric_series["map"]:
            c1.metric("MAP", f"{mean(metric_series['map']):.3f}")
            c2.metric("MAR", f"{mean(metric_series['mar']):.3f}")
            c3.metric("NDCG@5", f"{mean(metric_series['ndcg@5']):.3f}")
            c1.metric("P@5", f"{mean(metric_series['p@5']):.3f}")
            c2.metric("R@5", f"{mean(metric_series['r@5']):.3f}")
            c3.metric("F1@5", f"{mean(metric_series['f1@5']):.3f}")

        st.markdown("### Per-query results")
        st.dataframe(per_query_rows, use_container_width=True)


def render_ask_tab():
    st.subheader("Ask a question")
    query = st.text_input("Enter your question", value=st.session_state.last_query)
    top_k = st.slider("Number of chunks to include in answer generation", min_value=1, max_value=20, value=st.session_state.top_k)

    if st.button("Run query"):
        st.session_state.last_query = query
        st.session_state.top_k = top_k
        st.session_state.last_result = None
        st.session_state.last_error = None
        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/query",
                json={"query": query, "top_k": top_k},
                timeout=120,
            )
            response.raise_for_status()
            st.session_state.last_result = response.json()
        except Exception as e:
            st.session_state.last_error = str(e)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    result = st.session_state.last_result
    if result:
        st.markdown("### Answer")
        st.markdown(result.get("answer", ""))

        st.markdown("### Citations")
        st.json(result.get("citations", []))

        query_id = result.get("query_id")
        if query_id:
            st.caption(f"query_id: {query_id}")


def render_observability_tab():
    st.subheader("Observability")
    st.caption("Step-by-step trace for the most recent query.")

    result = st.session_state.last_result
    if not result:
        st.info("Run a query in Ask first.")
        return

    query_id = result.get("query_id")

    try:
        response = requests.get(
            f"{st.session_state.api_base_url}/logs/{query_id}",
            timeout=60,
        )
        response.raise_for_status()
        log = response.json()
    except Exception as e:
        st.error(str(e))
        return

    filters = log.get("filters", {})

    st.markdown("### 1. Query received")
    st.code(log.get("query", ""))

    st.markdown("### 2. Embedding generated")
    embedding = filters.get("query_embedding", [])
    st.caption(f"Dimensions: {len(embedding)}")
    if embedding:
        st.code(str([round(v, 6) for v in embedding[:8]]) + " ...")

    st.markdown("### 3. Vector retrieval")
    st.caption(f"Retrieved top {filters.get('retrieval_top_k', '?')} chunks")
    candidates = filters.get("retrieval_candidates", [])
    if candidates:
        st.dataframe(
            [{"rank": r["rank"], "foi_reference": r["foi_reference"],
              "page": r["page_number"], "score": round(r["score"], 4),
              "preview": r["content_preview"]} for r in candidates],
            use_container_width=True,
        )
    else:
        st.info("No retrieval candidates in log.")

    st.markdown("### 4. Reranking")
    st.caption(
        f"Model: {filters.get('reranker_model', '?')} | "
        f"{filters.get('rerank_input_count', '?')} in → "
        f"{filters.get('rerank_output_count', '?')} out"
    )
    reranked = filters.get("reranked_candidates", [])
    if reranked:
        st.dataframe(
            [{"rank": r["rank"], "foi_reference": r["foi_reference"],
              "page": r["page_number"], "preview": r["content_preview"]} for r in reranked],
            use_container_width=True,
        )
    else:
        st.info("No reranked candidates in log.")

    st.markdown("### 5. Prompt sent to generator")
    st.text_area("prompt", value=log.get("prompt_sent", ""),
                 height=280, disabled=True, label_visibility="collapsed")

    st.markdown("### 6. Answer")
    st.caption(f"Model: {log.get('model', '?')}")
    st.write(log.get("response", ""))



def render_evals_tab():
    st.subheader("Evals")
    st.caption("How well did the system find the right documents for your query?")

    st.info(
        "**How this works:** When you run a query, the system retrieves and ranks chunks from the database. "
        "This tab compares those results against a 'golden dataset' — a pre-built list of which chunks "
        "should be returned for known queries, and how relevant each one is (scored 0, 2, or 4). "
        "The scores below measure how well the system did."
    )

    dataset_path = st.text_input(
        "Golden dataset path",
        value="eval_data/hand_labelled_queries.csv",
        help="CSV file containing known good queries and their relevant chunks. Default is the hand-labelled dataset."
    )
    relevant_threshold = st.slider(
        "Relevant threshold (score >=)",
        min_value=0, max_value=4, value=2,
        help="Chunks scored at or above this number count as 'relevant'. Score 4 = very relevant, 2 = partially relevant, 0 = not relevant."
    )

    try:
        golden_df = load_golden_dataset(dataset_path)
    except Exception as e:
        st.error(f"Could not load golden dataset: {e}")
        return

    st.markdown("### Single query eval")
    result = st.session_state.last_result
    if not result:
        st.info("Run a query in Ask first to see single-query eval here. Or scroll down for batch eval.")
        render_batch_eval(golden_df, relevant_threshold)
        return

    query_id = result.get("query_id")

    try:
        response = requests.get(
            f"{st.session_state.api_base_url}/logs/{query_id}",
            timeout=60,
        )
        response.raise_for_status()
        log = response.json()
    except Exception as e:
        st.error(str(e))
        return

    filters = log.get("filters", {})

    reranked = filters.get("reranked_candidates", [])
    predicted_ids = [r["chunk_id"] for r in reranked]
    if not predicted_ids:
        predicted_ids = [str(c) for c in (log.get("retrieved_chunk_ids") or [])]

    if not predicted_ids:
        st.warning("No chunk IDs available for evaluation.")
        render_batch_eval(golden_df, relevant_threshold)
        return

    try:
        eval_result = evaluate_query_ranking(
            query_text=log.get("query", ""),
            predicted_chunk_ids=predicted_ids,
            golden_df=golden_df,
            ks=[1, 3, 5, 10],
            relevant_threshold=float(relevant_threshold),
        )
    except Exception as e:
        st.error(f"Eval failed: {e}")
        return

    if eval_result.matched_gold_rows == 0:
        st.warning(
            "This query wasn't found in the golden dataset. "
            "Only queries that were used to generate the golden dataset can be evaluated here. "
            "Try copying a query from `eval_data/golden_queries.md` into the Ask tab."
        )
        render_batch_eval(golden_df, relevant_threshold)
        return

    st.markdown("---")
    st.markdown("### Summary scores")
    st.caption(
        f"Query matched {eval_result.matched_gold_rows} rows in the golden dataset. "
        f"{eval_result.relevant_count} chunk(s) counted as relevant (score >= {relevant_threshold}). "
        f"System returned {eval_result.predicted_count} chunk(s)."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAP", f"{eval_result.average_precision:.3f}")
    c1.caption("**Mean Average Precision** — Did the system find the right chunks, and did it put them near the top? 1.0 = perfect, 0.0 = missed everything.")

    c2.metric("MAR", f"{eval_result.mar:.3f}")
    c2.caption("**Mean Average Recall** — Across different cutoff sizes, how consistently did it find the relevant chunks? 1.0 = always found them.")

    c3.metric("NDCG@5", f"{eval_result.ndcg_at_k.get(5, 0.0):.3f}")
    c3.caption("**Ranking quality in top 5** — Not just did it find the right chunk, but did it rank it first? 1.0 = most relevant chunk was ranked #1.")

    c4.metric("NDCG@10", f"{eval_result.ndcg_at_k.get(10, 0.0):.3f}")
    c4.caption("**Ranking quality in top 10** — Same as above but looking at a wider set of results.")

    st.markdown("---")
    st.markdown("### Precision / Recall / F1 at different cutoffs (k)")
    st.caption(
        "k = how many results we're looking at. "
        "k=1 means only the top result. k=10 means the top 10. "
        "Precision drops as k grows because more irrelevant chunks get included. "
        "Recall stays high if the system found the right chunk early."
    )

    st.dataframe(
        [
            {
                "k (top N results)": k,
                "precision": round(eval_result.precision_at_k.get(k, 0.0), 4),
                "recall": round(eval_result.recall_at_k.get(k, 0.0), 4),
                "f1": round(eval_result.f1_at_k.get(k, 0.0), 4),
                "ndcg": round(eval_result.ndcg_at_k.get(k, 0.0), 4),
            }
            for k in [1, 3, 5, 10]
        ],
        use_container_width=True,
    )

    with st.expander("What do these columns mean?"):
        st.markdown("""
**Precision** — Of the chunks the system returned, what fraction were actually relevant?
- e.g. precision@3 = 0.33 means 1 out of the top 3 chunks was relevant

**Recall** — Of all the relevant chunks that exist, what fraction did the system find?
- e.g. recall@5 = 1.0 means the system found all relevant chunks within its top 5

**F1** — A single combined score balancing precision and recall.
- High F1 means the system is both finding the right chunks AND not returning too much junk
- Falls as k grows because precision drops while recall stays flat

**NDCG** — Did the system rank the *most* relevant chunk highest?
- Unlike precision/recall, this rewards putting a score-4 chunk at rank 1 more than rank 5
- 1.0 = ideal ranking, 0.0 = completely wrong order

**A note on these scores:** This golden dataset was generated synthetically — queries were written by an AI specifically to match the chunks. Scores will be optimistically high as a result. These scores are most useful for spotting regressions (if a score drops after a code change, something broke) rather than as an absolute measure of quality.
        """)


    st.caption(
        f"Matched gold rows: {eval_result.matched_gold_rows} | "
        f"Relevant chunks (threshold >= {relevant_threshold}): {eval_result.relevant_count} | "
        f"Predicted chunks: {eval_result.predicted_count}"
    )

    render_batch_eval(golden_df, relevant_threshold)



def render_performance_tab():
    st.subheader("Performance")
    st.caption("Latency breakdown for the most recent query.")

    result = st.session_state.last_result
    if not result:
        st.info("Run a query in Ask first.")
        return

    query_id = result.get("query_id")

    try:
        response = requests.get(
            f"{st.session_state.api_base_url}/logs/{query_id}",
            timeout=60,
        )
        response.raise_for_status()
        log = response.json()
    except Exception as e:
        st.error(str(e))
        return

    filters = log.get("filters", {})

    st.markdown("### Pipeline steps")
    st.caption("Latency is not yet measured per-step — this shows what was retrieved.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval candidates", filters.get("rerank_input_count", "?"))
    c2.metric("After reranking", filters.get("rerank_output_count", "?"))
    c3.metric("Embedding dimensions", len(filters.get("query_embedding", [])))

    st.markdown("### Retrieval candidates")
    candidates = filters.get("retrieval_candidates", [])
    if candidates:
        st.dataframe(
            [{"rank": r["rank"], "foi_reference": r["foi_reference"],
              "page": r["page_number"], "score": round(r["score"], 4)} for r in candidates],
            use_container_width=True,
        )

    st.markdown("### Models used")
    st.dataframe([
        {"step": "Embedding", "model": "text-embedding-3-small", "provider": "OpenAI"},
        {"step": "Reranker", "model": filters.get("reranker_model", "?"), "provider": "Anthropic"},
        {"step": "Generator", "model": log.get("model", "?"), "provider": "Anthropic"},
    ], use_container_width=True)

    st.info(
        "Full latency per step (embed_ms, search_ms, rerank_ms, generate_ms) and token costs "
        "require timing instrumentation in the API — that's the next build step."
    )



def main():
    st.set_page_config(page_title="Camden FOI RAG", layout="wide")
    init_state()
    st.title("Camden FOI RAG")
    render_sidebar_stats()

    tab_ask, tab_obs, tab_evals, tab_perf = st.tabs([
        "Ask", "Observability", "Evals", "Performance"
    ])

    with tab_ask:
        render_ask_tab()
    with tab_obs:
        render_observability_tab()
    with tab_evals:
        render_evals_tab()
    with tab_perf:
        render_performance_tab()

if __name__ == "__main__":
    main()
