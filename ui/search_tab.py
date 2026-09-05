import gradio as gr
from config import TOP_K_DEFAULT
from search.engine import search_similar

def _run_search(image, top_k):
    if image is None:
        return [], "Upload a shoe photo to search."

    outcome = search_similar(image, top_k=int(top_k))
    status = outcome["status"]
    confidence = outcome["shoe_confidence"]
    results = outcome["results"]

    if status == "rejected":
        return [], (
            f"This doesn't look like a shoe (shoe-likelihood: {confidence:.0%}). "
            f"Try uploading a clearer photo of a single shoe, boot, sandal, or slipper."
        )

    if not results:
        return [], "No matches found -- is the catalog table populated? (see db/ingest.py)"

    gallery_items = [
        (r["image"], f"#{i + 1}  {r['similarity']:.1f}%\n{r['brand']}\n{r['category']} / {r['subcategory']}")
        for i, r in enumerate(results)
    ]

    if status == "low_confidence":
        summary = (
            f"Low-confidence matches (top similarity {results[0]['similarity']:.1f}%). "
            f"This image passed the shoe check but doesn't closely resemble anything in "
            f"the catalog -- results below may not be reliable."
        )
    else:
        summary = f"Found {len(results)} visually similar shoes."

    return gallery_items, summary

def build_search_tab():
    with gr.Column():
        gr.Markdown("### Upload a shoe photo to find visually similar products")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload shoe image", height=320)
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(1, 20, value=TOP_K_DEFAULT, step=1, label="Number of results")
                search_btn = gr.Button("Search", variant="primary")

        status = gr.Markdown()
        results_gallery = gr.Gallery(label="Similar shoes", columns=5, height="auto", object_fit="contain")

        search_btn.click(
            fn=_run_search,
            inputs=[image_input, top_k_slider],
            outputs=[results_gallery, status],
        )
        image_input.change(
            fn=_run_search,
            inputs=[image_input, top_k_slider],
            outputs=[results_gallery, status],
        )
