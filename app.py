import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils import embedding_functions as ef
import gradio as gr
import markdown
import ast
from sentence_transformers import SentenceTransformer


client = chromadb.PersistentClient(path="data/ICLR2026")
_ = SentenceTransformer("all-MiniLM-L6-v2")
# --- dynamic embedding selector ---
def get_collection(model_name: str, api_key: str):
    if model_name == "gemini-embedding-001":
        embedding_fn = ef.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
        COLLECTION_NAME = "Gemini"
    elif model_name == "all-MiniLM-L6-v2":
        embedding_fn = ef.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        COLLECTION_NAME = "MiniLM"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


# Scoring weights for plain search relevance calculation
TITLE_MATCH_BASE_SCORE = 0.5
TITLE_POSITION_WEIGHT = 0.3
TITLE_PROPORTION_WEIGHT = 0.2
ABSTRACT_MATCH_BASE_SCORE = 0.3
ABSTRACT_FREQUENCY_WEIGHT = 0.2
ABSTRACT_MAX_BONUS = 0.4


def plain_search(query_text, total_results=50):
    """
    Plain string matching search: searches titles first, then abstracts.
    Returns papers sorted by relevance (title matches first, then abstract matches).
    """
    if not query_text.strip():
        return None, "Please enter a query."
    
    try:
        # Get all documents from the default collection
        collection = get_collection("all-MiniLM-L6-v2", "")
        # Get all documents (ChromaDB max is typically around 100k, we'll get a large number)
        all_results = collection.get(limit=100000)
        
        docs = all_results["documents"]
        metas = all_results["metadatas"]
        ids = all_results["ids"]
        
        query_lower = query_text.lower().strip()
        
        title_matches = []
        abstract_matches = []
        
        for doc_id, doc, meta in zip(ids, docs, metas):
            title = meta.get("title", "Untitled")
            keywords_raw = meta.get("keywords", "")
            pdf = meta.get("pdf", "")
            bibtex = meta.get("_bibtex", "")
            
            try:
                if isinstance(keywords_raw, str) and keywords_raw.strip().startswith("["):
                    keywords = ast.literal_eval(keywords_raw)
                elif isinstance(keywords_raw, list):
                    keywords = keywords_raw
                else:
                    keywords = [str(keywords_raw)]
            except Exception:
                keywords = [str(keywords_raw)]
            
            record = {
                "title": title,
                "keywords": keywords,
                "pdf": pdf,
                "abstract_md": doc.strip() if doc else "",
                "bibtex": bibtex,
                "similarity": 0.0  # Will be updated based on match quality
            }
            
            # Check for title match
            if query_lower in title.lower():
                # Calculate relevance score: higher if match appears earlier and covers more of title
                title_lower = title.lower()
                match_position = title_lower.find(query_lower)
                position_score = 1.0 - (match_position / max(len(title_lower), 1))
                proportion_score = len(query_lower) / max(len(title_lower), 1)
                record["similarity"] = round(
                    TITLE_MATCH_BASE_SCORE + TITLE_POSITION_WEIGHT * position_score + TITLE_PROPORTION_WEIGHT * proportion_score, 
                    4
                )
                title_matches.append(record)
            # Check for abstract match (only if not already in title matches)
            elif query_lower in doc.lower() if doc else False:
                # Lower score for abstract matches, bonus for multiple occurrences
                abstract_lower = doc.lower()
                match_count = abstract_lower.count(query_lower)
                record["similarity"] = round(
                    ABSTRACT_MATCH_BASE_SCORE + min(ABSTRACT_FREQUENCY_WEIGHT * match_count, ABSTRACT_MAX_BONUS), 
                    4
                )
                abstract_matches.append(record)
        
        # Sort each group by similarity score (descending)
        title_matches.sort(key=lambda x: x["similarity"], reverse=True)
        abstract_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Combine: title matches first, then abstract matches
        all_matches = title_matches + abstract_matches
        
        # Limit to requested number of results
        limited_matches = all_matches[:int(total_results)]
        
        if not limited_matches:
            return [], None
        
        return limited_matches, None
    except Exception as e:
        return None, f"Error: {e}"


def query_db(model_name, api_key, query_text, total_results=50):
    if not query_text.strip():
        return None, "Please enter a query."
    if model_name == "gemini-embedding-001" and not api_key.strip():
        return None, "Please enter your Gemini API key."

    try:
        collection = get_collection(model_name, api_key)
        results = collection.query(query_texts=[query_text], n_results=int(total_results))
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        dists = results["distances"][0]

        records = []
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            title = meta.get("title", "Untitled")
            keywords_raw = meta.get("keywords", "")
            pdf = meta.get("pdf", "")
            bibtex = meta.get("_bibtex", "")
            similarity = round(1 - dist, 4) if dist <= 1 else round(dist, 4)

            try:
                if isinstance(keywords_raw, str) and keywords_raw.strip().startswith("["):
                    keywords = ast.literal_eval(keywords_raw)
                elif isinstance(keywords_raw, list):
                    keywords = keywords_raw
                else:
                    keywords = [str(keywords_raw)]
            except Exception:
                keywords = [str(keywords_raw)]

            records.append({
                "title": title,
                "keywords": keywords,
                "pdf": pdf,
                "abstract_md": doc.strip(),
                "bibtex": bibtex,
                "similarity": similarity
            })
        return records, None
    except Exception as e:
        return None, f"Error: {e}"


def render_page(records, page, per_page=10):
    if not records:
        return "<p>No results to show.</p>"
    total_pages = (len(records) - 1) // per_page + 1
    page = max(1, min(page, total_pages))
    start, end = (page - 1) * per_page, min(page * per_page, len(records))
    html = ""
    for r in records[start:end]:
        abstract_html = markdown.markdown(r["abstract_md"], extensions=["fenced_code", "tables"])
        keyword_html = " ".join([
            f"<span class='keyword'>{k.strip().title()}</span>"
            for k in r["keywords"] if k and isinstance(k, str)
        ])
        html += f"""
        <div class='paper-card'>
            <h3>{r['title']}</h3>
            <p><b>Affinity Score:</b> {r['similarity']}</p>
            <p><b>Keywords:</b> {keyword_html}</p>
            <p><b>PDF:</b> <a href='{r['pdf']}' target='_blank'>{r['pdf']}</a></p>
            <details><summary>Show Abstract</summary>
              <div class='abstract markdown-body'>{abstract_html}</div>
            </details>
            <details><summary>Show BibTeX</summary>
              <div class='bibtex'><pre>{r['bibtex']}</pre></div>
            </details>
        </div>"""
    html += f"<div class='page-info'>Page {page} / {total_pages}</div>"
    return html


# --- UI ---
with gr.Blocks(title="ICLR 2026 Paper Search") as demo:
    gr.Markdown("## ICLR 2026 Paper Search")
    gr.Markdown("Semantic search over ICLR 2026 submissions.")

    with gr.Accordion("Search Options", open=True) as search_box:
        search_mode = gr.Radio(
            label="Search Mode",
            choices=["Semantic Search", "Plain Search"],
            value="Semantic Search",
            info="Semantic Search uses embeddings, Plain Search uses direct string matching"
        )
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Embedding Model",
                choices=["gemini-embedding-001", "all-MiniLM-L6-v2"],
                value="all-MiniLM-L6-v2",
                interactive=True
            )
            api_key_box = gr.Textbox(
                label="API Key (required for some embedding models)",
                type="password",
                placeholder="Enter Gemini API key",
                visible=False
            )
        total_results = gr.Number(label="Total number of results to retrieve", value=50, precision=0)
        query = gr.Textbox(label="Query (abstract of a paper)", placeholder="e.g., diffusion models in text-to-image generation", lines=2)
        search_btn = gr.Button("Search")

    results_box = gr.HTML("<p>Results will appear here.</p>")
    records_state, page_state = gr.State([]), gr.State(1)

    # hide/show api key dynamically
    def toggle_key(model_name):
        return gr.update(visible=(model_name == "gemini-embedding-001"))
    model_dropdown.change(toggle_key, inputs=model_dropdown, outputs=api_key_box)

    # hide/show embedding model options based on search mode
    def toggle_embedding_options(mode, current_model):
        if mode == "Plain Search":
            return gr.update(visible=False), gr.update(visible=False)
        else:
            # Show model dropdown, API key only if Gemini is selected
            api_key_visible = (current_model == "gemini-embedding-001")
            return gr.update(visible=True), gr.update(visible=api_key_visible)
    search_mode.change(toggle_embedding_options, inputs=[search_mode, model_dropdown], outputs=[model_dropdown, api_key_box])

    def on_search(mode, model, key, q, total_res):
        if mode == "Plain Search":
            recs, err = plain_search(q, total_res)
        else:
            recs, err = query_db(model, key, q, total_res)
        
        if err:
            return gr.update(open=True), f"<p style='color:red;'>{err}</p>", [], 1
        return gr.update(open=False), render_page(recs, 1), recs, 1

    search_btn.click(on_search,
        inputs=[search_mode, model_dropdown, api_key_box, query, total_results],
        outputs=[search_box, results_box, records_state, page_state])

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")

    def change_page(records, page, direction):
        new_page = page + direction
        return render_page(records, new_page), new_page

    prev_btn.click(change_page,
        inputs=[records_state, page_state, gr.Number(value=-1, visible=False)],
        outputs=[results_box, page_state])
    next_btn.click(change_page,
        inputs=[records_state, page_state, gr.Number(value=1, visible=False)],
        outputs=[results_box, page_state])

    gr.HTML("""
    <style>
    #component-1{max-width:950px;margin:auto;}
    .paper-card{background:#fff;border-radius:10px;padding:16px;margin-bottom:18px;
                box-shadow:0 2px 8px rgba(0,0,0,0.1);}
    .keyword{display:inline-block;background:#e8f1ff;color:#003d99;border-radius:6px;
             padding:2px 8px;margin:2px;font-size:13px;}
    details summary{cursor:pointer;font-weight:600;color:#0066cc;}
    .abstract,.bibtex{background:#f8f8f8;padding:10px;border-radius:6px;margin-top:8px;}
    pre{background:#f9f9f9;border:1px solid #ddd;padding:8px;border-radius:4px;overflow-x:auto;}
    .page-info{text-align:center;font-weight:bold;margin-top:10px;}
    </style>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
    <script>
      const obs=new MutationObserver(()=>{if(window.renderMathInElement)
        renderMathInElement(document.body,{delimiters:[
          {left:'$$',right:'$$',display:true},
          {left:'$',right:'$',display:false}]});});
      obs.observe(document.body,{childList:true,subtree:true});
    </script>
    """)

demo.launch()
