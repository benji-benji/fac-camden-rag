import streamlit as st
import requests
import sys
from pathlib import Path
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
    st.caption("Retrieval quality metrics against the golden dataset.")

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

    # get predicted chunk ids in ranked order
    reranked = filters.get("reranked_candidates", [])
    predicted_ids = [r["chunk_id"] for r in reranked]
    if not predicted_ids:
        predicted_ids = [str(c) for c in (log.get("retrieved_chunk_ids") or [])]

    if not predicted_ids:
        st.warning("No chunk IDs available for evaluation.")
        return

    dataset_path = st.text_input(
        "Golden dataset path",
        value="eval_data/synthetic_chunk_query_pairs.csv",
    )
    relevant_threshold = st.slider(
        "Relevant threshold (score >=)", min_value=0, max_value=4, value=2
    )

    try:
        golden_df = load_golden_dataset(dataset_path)
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
        st.warning("Query not found in golden dataset — run generate_ground_truth.py first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAP", f"{eval_result.average_precision:.3f}")
    c2.metric("MAR", f"{eval_result.mar:.3f}")
    c3.metric("NDCG@5", f"{eval_result.ndcg_at_k.get(5, 0.0):.3f}")
    c4.metric("NDCG@10", f"{eval_result.ndcg_at_k.get(10, 0.0):.3f}")

    st.markdown("### Precision / Recall / F1 by K")
    st.dataframe(
        [
            {
                "k": k,
                "precision": round(eval_result.precision_at_k.get(k, 0.0), 4),
                "recall": round(eval_result.recall_at_k.get(k, 0.0), 4),
                "f1": round(eval_result.f1_at_k.get(k, 0.0), 4),
                "ndcg": round(eval_result.ndcg_at_k.get(k, 0.0), 4),
            }
            for k in [1, 3, 5, 10]
        ],
        use_container_width=True,
    )

    st.caption(
        f"Matched gold rows: {eval_result.matched_gold_rows} | "
        f"Relevant chunks (threshold >= {relevant_threshold}): {eval_result.relevant_count} | "
        f"Predicted chunks: {eval_result.predicted_count}"
    )



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
