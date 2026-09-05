import gradio as gr

from config import CATALOG_PAGE_SIZE
from search.engine import get_catalog_page, get_filter_options

ALL_OPTION = "All"

def _load_page(page, category, subcategory, brand):
    category = None if category in (None, ALL_OPTION) else category
    subcategory = None if subcategory in (None, ALL_OPTION) else subcategory
    brand = None if brand in (None, ALL_OPTION) else brand

    items, total = get_catalog_page(
        page=page, page_size=CATALOG_PAGE_SIZE,
        category=category, subcategory=subcategory, brand=brand,
    )
    gallery_items = [
        (item["image"], f"{item['brand']}\n{item['category']} / {item['subcategory']}")
        for item in items
    ]
    max_page = max(1, -(-total // CATALOG_PAGE_SIZE))  # ceil division
    status = f"Page {int(page)} of {max_page} -- {total} products total"
    return gallery_items, status


def build_catalog_tab():
    categories, subcategories, brands = get_filter_options()

    with gr.Column():
        gr.Markdown("### Product catalog")

        with gr.Row():
            category_dd = gr.Dropdown([ALL_OPTION] + categories, value=ALL_OPTION, label="Category")
            subcategory_dd = gr.Dropdown([ALL_OPTION] + subcategories, value=ALL_OPTION, label="Subcategory")
            brand_dd = gr.Dropdown([ALL_OPTION] + brands, value=ALL_OPTION, label="Brand")
            page_num = gr.Number(value=1, precision=0, label="Page")
            load_btn = gr.Button("Load", variant="primary")

        status = gr.Markdown()
        catalog_gallery = gr.Gallery(label="Catalog", columns=6, height="auto", object_fit="contain")

        inputs = [page_num, category_dd, subcategory_dd, brand_dd]
        outputs = [catalog_gallery, status]

        load_btn.click(fn=_load_page, inputs=inputs, outputs=outputs)
        for dd in (category_dd, subcategory_dd, brand_dd):
            dd.change(fn=lambda c, s, b: _load_page(1, c, s, b),
                      inputs=[category_dd, subcategory_dd, brand_dd],
                      outputs=outputs)

    return _load_page, inputs, outputs
