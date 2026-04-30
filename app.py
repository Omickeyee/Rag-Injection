import streamlit as st
import torch
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings

from docs import build_dataset
from rag import RAGEnv
from defense import chunk_scanner, safety_reranker, prepare_defended_docs, get_trust_score


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_QUERIES = [
    "What is Python?",
    "Tell me about the Great Wall of China",
    "Where is the Eiffel Tower located?",
]


CARD_CSS = """
<style>
.doc-card {
    border: 1px solid #d0d7de;
    border-radius: 14px;
    padding: 14px 16px;
    background: #f8fafc;
    margin-bottom: 12px;
}
.doc-card.malicious {
    border-color: #dc2626;
    background: #fef2f2;
}
.doc-card.benign {
    border-color: #16a34a;
    background: #f0fdf4;
}
.doc-card.hard-negative {
    border-color: #d97706;
    background: #fffbeb;
}
.doc-card.filtered {
    border-color: #7c3aed;
    background: #f5f3ff;
}
.doc-row {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    margin: 4px 0;
    line-height: 1.35;
}
.doc-key {
    min-width: 112px;
    font-weight: 700;
    color: #0f172a;
}
.doc-value {
    color: #1e293b;
    word-break: break-word;
}
</style>
"""


def get_device():
    device = "cpu"
    device_index = -1
    if torch.backends.mps.is_available():
        device = "mps"
        device_index = "mps"
    if torch.cuda.is_available():
        device = "cuda"
        device_index = 0
    return device, device_index


@st.cache_resource(show_spinner=False)
def load_env(model_name, dataset_size, attack_rate, seed, defense_llm):
    device, device_index = get_device()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    rag_model = pipeline(
        "text-generation",
        model=model_name,
        device=device_index,
        dtype=torch.float32,
    )
    documents = build_dataset(
        n=dataset_size,
        attack_rate=attack_rate,
        seed=seed,
        focus_queries=DEFAULT_QUERIES,
        preferred_attack_type="goal_hijacking",
    )
    env = RAGEnv(docs=documents, llm=rag_model, embeddings=embeddings)
    return env, defense_llm, device


def label_for_retrieval_item(retrieval_item):
    source_kind = retrieval_item.get("source_kind")
    if retrieval_item.get("poisoned"):
        return "malicious"
    if source_kind == "hard_negative":
        return "hard_negative"
    return "benign"


def run_pipeline(env, query, defense_llm):
    old_response, old_retrieval = env.query(query)
    raw_docs_for_defense = []
    original_by_id = {}
    initial_docs = []

    for i, retrieval_item in enumerate(old_retrieval):
        doc_text = retrieval_item["content"]
        label = label_for_retrieval_item(retrieval_item)
        metadata = {
            "source_kind": retrieval_item.get("source_kind"),
            "topic": retrieval_item.get("topic"),
            "attack": retrieval_item.get("attack"),
            "poisoned": retrieval_item.get("poisoned"),
            "relevance_score": 1.0 - (i / max(len(old_retrieval), 1)),
        }
        suspicion_score = chunk_scanner(doc_text, defense_llm)
        trust_score = get_trust_score(label, metadata)
        initial_docs.append(
            {
                **retrieval_item,
                "label": label,
                "suspicion_score": suspicion_score,
                "trust_score": trust_score,
            }
        )
        raw_docs_for_defense.append((retrieval_item["doc_id"], doc_text, label, metadata))
        original_by_id[retrieval_item["doc_id"]] = retrieval_item

    reranked_docs = safety_reranker(raw_docs_for_defense, defense_llm)
    reranked_triplets = [
        (item["doc_id"], item["text"], item["label"])
        for item in reranked_docs
    ]
    analyses, defended_docs = prepare_defended_docs(reranked_triplets, defense_llm=defense_llm)

    defended_view = []
    defended_retrieval = []
    for (doc_id, _, original_label, analysis), (defended_doc_id, defended_text, defended_label), reranked_item in zip(
        analyses,
        defended_docs,
        reranked_docs,
    ):
        original = original_by_id[defended_doc_id]
        defended_view.append(
            {
                "doc_id": doc_id,
                "label": original_label,
                "filtered": analysis["suspicious"],
                "suspicion_score": analysis.get("suspicion_score"),
                "rerank_score": reranked_item["final_score"],
                "safety_score": reranked_item["safety_score"],
                "trust_score": reranked_item["trust_score"],
                "topic": original.get("topic"),
                "attack": original.get("attack"),
                "content": defended_text,
            }
        )
        if defended_label != "filtered":
            defended_retrieval.append(
                {
                    "doc_id": defended_doc_id,
                    "topic": original.get("topic"),
                    "attack": original.get("attack"),
                    "poisoned": original.get("poisoned"),
                    "source_kind": original.get("source_kind"),
                    "content": defended_text,
                }
            )

    if defended_retrieval:
        new_response = env.generate_from_retrieval(query, defended_retrieval, defended=True)
    else:
        new_response = "No safe retrieved documents remained after filtering."

    return {
        "old_response": old_response,
        "new_response": new_response,
        "initial_docs": initial_docs,
        "defended_docs": defended_view,
    }


def render_doc_card(item, rows, tone):
    row_html = "".join(
        f"<div class='doc-row'><div class='doc-key'>{label}</div><div class='doc-value'>{value}</div></div>"
        for label, value in rows
    )
    st.markdown(
        f"<div class='doc-card {tone}'>{row_html}</div>",
        unsafe_allow_html=True,
    )
    st.code(item["content"], language="text")


st.set_page_config(page_title="RAG Injection Defense", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)

st.title("RAG Injection Defense")
st.caption("Compare raw retrieved context against reranked and filtered defended context.")

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    defense_llm = st.toggle("Use learned detector", value=True)
    dataset_size = st.slider("Dataset size", min_value=6, max_value=24, value=12, step=2)
    attack_rate = st.slider("Attack rate", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    seed = st.number_input("Seed", min_value=0, value=7, step=1)
    query_mode = st.radio("Query mode", ["Preset", "Custom"], index=0)

if query_mode == "Preset":
    query = st.selectbox("Query", DEFAULT_QUERIES)
else:
    query = st.text_input("Custom query", value=DEFAULT_QUERIES[0])

run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)

if run_clicked:
    if not query.strip():
        st.error("Enter a query first.")
        st.stop()

    with st.spinner("Loading model, corpus, and retrieval pipeline..."):
        env, defense_flag, device = load_env(
            model_name=model_name.strip(),
            dataset_size=dataset_size,
            attack_rate=attack_rate,
            seed=int(seed),
            defense_llm=int(defense_llm),
        )
    
    if int(defense_llm) == 1:
        st.info(f"Running on {device} with both manual and LLM defense")
    else:
        st.info(f"Running on {device} with manual defense and no defense LLM")

    with st.spinner("Running raw and defended pipelines..."):
        result = run_pipeline(env, query.strip(), int(defense_llm))

    st.subheader("Initial Retrieved Documents")
    raw_cols = st.columns(len(result["initial_docs"]))
    for col, item in zip(raw_cols, result["initial_docs"]):
        tone = item["label"].replace("_", "-")
        with col:
            render_doc_card(
                item,
                [
                    ("doc_id", item["doc_id"]),
                    ("topic", item["topic"]),
                    ("attack", item["attack"]),
                    ("label", item["label"]),
                    ("suspicion", f"{item['suspicion_score']:.3f}"),
                    ("trust", f"{item['trust_score']:.3f}"),
                ],
                tone,
            )

    st.subheader("Responses")
    response_cols = st.columns(2)
    with response_cols[0]:
        st.markdown("**Old response**")
        st.write(result["old_response"])
    with response_cols[1]:
        st.markdown("**New response**")
        st.write(result["new_response"])

    st.subheader("Defended Documents After Reranking and Filtering")
    defended_cols = st.columns(len(result["defended_docs"]))
    for col, item in zip(defended_cols, result["defended_docs"]):
        with col:
            tone = "filtered" if item["filtered"] else item["label"].replace("_", "-")
            render_doc_card(
                item,
                [
                    ("doc_id", item["doc_id"]),
                    ("topic", item["topic"]),
                    ("attack", item["attack"]),
                    ("label", item["label"]),
                    ("filtered", str(item["filtered"])),
                    ("rerank score", f"{item['rerank_score']:.3f}"),
                    ("safety", f"{item['safety_score']:.3f}"),
                    ("trust", f"{item['trust_score']:.3f}"),
                    ("suspicion", f"{item['suspicion_score']:.3f}"),
                ],
                tone,
            )
