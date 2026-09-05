# Shoe Visual Search

Upload a photo of a shoe and instantly find visually similar products from a catalog
of **8,783 shoes** (UT-Zappos50K dataset). The system uses a **fine-tuned CLIP visual
encoder** to embed images, and **PostgreSQL + pgvector** to run fast nearest-neighbor
search — served through a two-tab **Gradio** web app.

## Features

- **Visual Search** — upload any shoe photo (or take one with your camera) and get
  the top-K visually similar products, ranked by similarity score.
- **Product Catalog** — browse the full catalog with category / subcategory / brand
  filters and pagination.
- **Shoe gate** — a zero-shot CLIP pre-check rejects non-shoe images before any
  costly search happens, and flags low-confidence matches with a warning.
- **Privacy-friendly** — your query photo is processed in memory and **never stored**.
- **Stateless & lightweight** — vectors and images both live in Postgres, so the app
  itself has no heavy state and deploys cleanly to Hugging Face Spaces (GPU).

## How it works

```
              ┌──────────────────────────────┐
   image ───▶ │ 1. Shoe gate (zero-shot CLIP) │  rejected → "this isn't a shoe" 
              └──────────────┬───────────────┘
                             │ passed
                             ▼
              ┌──────────────────────────────┐
              │ 2. Fine-tuned CLIP encoder   │ ──▶ 512-dim L2-normalized embedding
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ 3. pgvector cosine (<=>) scan│ ──▶ top-K similar products
              └──────────────────────────────┘
```

1. **Shoe gate** (`model/gate.py`): an un-fine-tuned CLIP copy does zero-shot
   image-text classification between "shoe" and "not shoe" prompts.
2. **Embedding** (`model/encoder.py`): the fine-tuned CLIP `ViT-B/32` *visual* encoder
   converts the query image into a 512-dimensional embedding.
3. **Search** (`search/engine.py`): a pgvector **cosine-distance** query (`<=>`) finds
   the nearest neighbors against the fine-tuned embeddings of all 8,783 catalog items.

### Model fine-tuning

The visual encoder was fine-tuned on shoe subcategories using **supervised contrastive
loss (SupCon)**, training only the last 2 transformer blocks, layer-norm, and the final
projection (~14.5M of 151M params, ~9.6%). Full pipeline and evaluation live in the
[`shoe_visual_search_finetuned.ipynb`](shoe_visual_search_finetuned.ipynb) notebook.

> **Evaluation (leak-free held-out validation set):**
>
> | Metric | Score |
> |---|---|
> | Recall@1 | **81.91%** |
> | Recall@5 | **94.91%** |
> | Recall@10 | **97.26%** |

## Project layout

```
shoe-visual-search/
├── app.py                       # Gradio entrypoint (two tabs)
├── config.py                    # env-driven settings (DB, model, thresholds)
├── requirements.txt             # Python dependencies
├── .env                         # environment config (DATABASE_URL, ...)
├── shoe_visual_search_finetuned.ipynb  # fine-tuning + export notebook (Colab)
│
├── db/                          # Postgres + pgvector layer
│   ├── schema.sql               # pgvector extension + products table DDL
│   ├── connection.py            # psycopg2 + pgvector connection helpers
│   └── ingest.py                # one-time loader: notebook export → Postgres
│
├── model/                       # ML inference layer
│   ├── encoder.py               # fine-tuned CLIP visual encoder + embed_image()
│   ├── gate.py                  # zero-shot "is this a shoe?" gate
│   └── weights/
│       └── clip_visual_finetuned.pt   # fine-tuned ViT-B/32 weights (~335 MB)
│
├── search/
│   └── engine.py                # pgvector similarity search + catalog browsing
│
├── ui/
│   ├── search_tab.py            # Page 1: Visual Search
│   └── catalog_tab.py           # Page 2: Product Catalog
│
└── data/
    └── export_for_app/          # notebook Section 17 export (gitignored)
        ├── catalog_export.json  # 8,783 records: metadata + embeddings
        └── catalog_images/      # 8,783 JPEGs (one per product)
```

## Getting started

### Prerequisites

- Python 3.12
- A PostgreSQL database with the **pgvector** extension enabled
  (free tiers: [Neon](https://neon.tech), [Supabase](https://supabase.com),
  [Render](https://render.com), or self-hosted)

### 1. Configure the environment

Create a `.env` file inside the project root with at least the `DATABASE_URL` set:

```bash
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>?sslmode=require
```

Available settings (all optional — defaults shown) are defined in `config.py`:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | — (required) | Postgres/pgvector connection string |
| `CLIP_MODEL_NAME` | `ViT-B/32` | CLIP model variant |
| `FINETUNED_WEIGHTS_PATH` | `model/weights/clip_visual_finetuned.pt` | fine-tuned weights |
| `EMBEDDING_DIM` | `512` | embedding dimension (must match `vector(N)` in the schema) |
| `TOP_K_DEFAULT` | `5` | default number of visual-search results |
| `SHOE_GATE_THRESHOLD` | `0.5` | min shoe-likelihood to pass the gate |
| `LOW_CONFIDENCE_SIMILARITY` | `55` | warn below this % similarity |
| `CATALOG_PAGE_SIZE` | `24` | items per catalog page |

### 2. Install dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `requirements.txt` installs CLIP straight from GitHub (`git+https://github.com/openai/CLIP.git`),
> which needs a compiler toolchain on the machine installing it (e.g. CUDA build tools
> on Windows, or a Linux space).

### 3. Ingest the catalog into Postgres (one-time)

The fine-tuned embeddings and catalog images must be loaded into Postgres once:

```bash
python -m db.ingest --export-dir data/export_for_app
```

This creates the `products` table (metadata + image `BYTEA` + `VECTOR(512)` embedding),
loads all 8,783 rows, and builds the `ivfflat` ANN index afterwards.

If `data/export_for_app/` is missing, run Sections 1–17 of the notebook and copy its
`export_for_app/` output here (see `data/README.md`).

### 4. Run the app

```bash
python app.py
```

Open the printed local URL:

- **Visual Search tab** — upload a shoe photo, adjust *Number of results*, then hit *Search*.
- **Product Catalog tab** — filter by category / subcategory / brand and paginate through all products.

## Deploying to Hugging Face Spaces

1. Create a new Space with SDK = **Gradio**.
2. Push everything in this folder **except** `data/export_for_app/` (large; only needed
   for the one-time ingestion — already in `.gitignore`).
3. Commit `model/weights/clip_visual_finetuned.pt` (use **git-lfs** — it's ~335 MB;
   Spaces supports LFS).
4. Add `DATABASE_URL` (and any overridden settings) under **Settings → Repository secrets**.
5. Run `python -m db.ingest --export-dir data/export_for_app` **once from your local
   machine** pointed at the same `DATABASE_URL` (ingestion only needs to happen once —
   the data lives in Postgres, not in the Space's filesystem).
6. Push. The Space installs `requirements.txt` and runs `app.py` automatically.

## Handling non-shoe uploads

The fine-tuned encoder was only ever trained to separate shoe *subcategories* — it
never saw a non-shoe image, so an unrelated photo (furniture, a face, a pet) would
still land somewhere in shoe-embedding space and produce deceptively confident results.
`model/gate.py` fixes this with a pre-check: a second, un-fine-tuned CLIP copy runs
zero-shot classification against a bank of "shoe" vs. "not shoe" text prompts *before*
retrieval. `search/engine.py` then returns one of three statuses:

- **`rejected`** — the gate says it isn't shoe-related; the UI shows a message and no
  DB query happens.
- **`low_confidence`** — passes the gate, but the best match is still below
  `LOW_CONFIDENCE_SIMILARITY`; results are shown with a warning banner.
- **`ok`** — normal case, results as expected.

Both thresholds live in `.env` / `config.py`. Tune them by running a handful of real
shoe photos and a handful of obviously unrelated photos, watching the printed
`shoe_confidence` / similarity values, then setting the thresholds between the two
clusters you observe.

## Notes & decisions

- Embeddings are **L2-normalized**, so search uses pgvector's **cosine** distance
  operator (`<=>`) — matching how the notebook evaluated Recall@K.
- `model/encoder.py` loads the same `ViT-B/32` backbone as the notebook and drops in the
  fine-tuned **visual** encoder weights only; the text tower is unused at runtime.
- The `ivfflat` ANN index is built in `db/ingest.py` *after* the data is loaded (not in
  `schema.sql`), because ivfflat clusters from whatever rows exist at `CREATE INDEX`
  time — building it on an empty table would badly hurt recall.
- If you regenerate embeddings with a different CLIP variant, update `CLIP_MODEL_NAME`,
  `EMBEDDING_DIM`, and the `vector(N)` dimension in `db/schema.sql` to match.

## Roadmap ideas

- HNSW index instead of ivfflat for better latency/recall on larger catalogs.
- Text-based search alongside visual search (the CLIP text tower is already available).
- Approximate query-per-second metrics and latency benchmarks.

## License

This project is provided as-is for demonstration purposes. The embedded catalog data is
derived from the UT-Zappos50K dataset; the model weights are a fine-tuned OpenAI CLIP
`ViT-B/32` visual encoder.
