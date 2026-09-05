import spaces
import gradio as gr
from ui.search_tab import build_search_tab
from ui.catalog_tab import build_catalog_tab


def build_app():
    with gr.Blocks(title="Shoe Visual Search") as demo:
        gr.Markdown("# Shoe Visual Search")

        with gr.Tabs():
            with gr.Tab("Visual Search"):
                build_search_tab()
            with gr.Tab("Product Catalog"):
                load_fn, inputs, outputs = build_catalog_tab()
                demo.load(fn=load_fn, inputs=inputs, outputs=outputs)

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(primary_hue="yellow"))