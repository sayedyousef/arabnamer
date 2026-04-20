"""Gradio demo for arabnamer — live on Hugging Face Spaces.

Three tabs:
  1. Transliterate  — English name -> Arabic name (XGBoost / rules / hybrid engines)
  2. Similarity     — two Arabic strings -> lenient similarity score
  3. Batch          — paste many English names -> CSV-style output

Runs fully offline inside the Space container. No external API calls.
"""
from __future__ import annotations

import csv
import io

import gradio as gr

from arabnamer import Transliterator, similarity

# Lazy singletons — load once, reuse for all requests
_XGB = Transliterator(engine="model", threshold=85)
_RULES = Transliterator(engine="rules", threshold=85)
_HYBRID = Transliterator(engine="hybrid", threshold=85)


def _get_engine(name: str) -> Transliterator:
    return {"model (XGBoost)": _XGB, "rules (deterministic)": _RULES, "hybrid": _HYBRID}[name]


def translit_single(name_en: str, engine: str, reference: str | None, threshold: int) -> tuple[str, str, str]:
    """Transliterate a single English name.

    Returns: (arabic, score_display, details_markdown)
    """
    if not name_en or not name_en.strip():
        return "", "—", "Enter an English name above."

    t = _get_engine(engine)
    t.threshold = threshold
    ref = reference.strip() if reference and reference.strip() else None
    r = t.translit(name_en, reference=ref)

    if ref:
        score_display = f"{r.score:.1f} / 100" + ("  ✅ accepted" if r.accepted else "  ❌ below threshold")
    else:
        score_display = "—  (no reference supplied)"

    details = f"""**Engine used:** `{r.engine}`

**Input:** `{r.input}`
**Predicted Arabic:** `{r.arabic}`
**Reference:** {f'`{r.reference}`' if r.reference else '_not provided_'}

{'**Score:** ' + str(r.score) + ' (threshold ' + str(threshold) + ')' if ref else ''}
"""
    return r.arabic, score_display, details


def similarity_pair(a: str, b: str, threshold: int) -> tuple[str, str, str]:
    """Score Arabic-to-Arabic similarity with the lenient normalizer."""
    if not a or not b:
        return "—", "—", "Enter two Arabic strings above."

    passed, score = similarity(a, b, threshold=threshold)
    verdict = "✅ match" if passed else "❌ not a match (below threshold)"

    # Normalized forms (for debugging / transparency)
    from arabnamer.scoring import normalize_arabic
    na, nb = normalize_arabic(a), normalize_arabic(b)
    details = f"""**Input A:** `{a}`
**Input A (normalized):** `{na}`

**Input B:** `{b}`
**Input B (normalized):** `{nb}`

**Score:** {score} / 100  (threshold: {threshold})
"""
    return verdict, f"{score} / 100", details


def batch_transliterate(input_text: str, engine: str) -> tuple[str, str]:
    """Run a list of English names through the selected engine.

    Input: one name per line.
    Output: markdown table + CSV string.
    """
    if not input_text or not input_text.strip():
        return "Paste English names above (one per line).", ""

    names = [line.strip() for line in input_text.splitlines() if line.strip()]
    t = _get_engine(engine)

    rows = [t.translit(n) for n in names]

    # Markdown table
    md_lines = ["| English | Arabic | Engine |", "|---|---|---|"]
    for r in rows:
        md_lines.append(f"| `{r.input}` | `{r.arabic}` | `{r.engine}` |")
    md = "\n".join(md_lines)

    # CSV string
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["name_en", "name_ar", "engine"])
    for r in rows:
        w.writerow([r.input, r.arabic, r.engine])

    return md, buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="arabnamer — Arabic name transliteration & similarity") as demo:
    gr.Markdown(
        """
# arabnamer — Arabic name transliteration & similarity

**Offline** English → Arabic name transliteration and Arabic-to-Arabic fuzzy matching.
No LLM, no external API, names never leave this Space. Bundled with a 38 MB pruned
XGBoost model trained on 22,798 English-Arabic name pairs.

**Install on your own machine:**
```bash
pip install arabnamer
```

🔗 [GitHub](https://github.com/sayedyousef/arabnamer)  ·
🔗 [PyPI](https://pypi.org/project/arabnamer/)  ·
🔗 [Model](https://huggingface.co/Sayedyousef/arabnamer-xgboost)  ·
🔗 [Dataset](https://huggingface.co/datasets/Sayedyousef/arabic-name-pairs)
        """
    )

    with gr.Tab("1. Transliterate"):
        gr.Markdown("### English → Arabic")
        with gr.Row():
            with gr.Column():
                name_in = gr.Textbox(
                    label="English name",
                    placeholder="Mohammed Ali",
                    lines=1,
                )
                engine_pick = gr.Radio(
                    ["model (XGBoost)", "rules (deterministic)", "hybrid"],
                    value="model (XGBoost)",
                    label="Engine",
                )
                ref_in = gr.Textbox(
                    label="Reference Arabic (optional — enables scoring)",
                    placeholder="محمد علي",
                    lines=1,
                )
                thresh_t = gr.Slider(
                    minimum=0, maximum=100, value=85, step=1,
                    label="Pass threshold (lenient score)",
                )
                btn_t = gr.Button("Transliterate", variant="primary")

            with gr.Column():
                ar_out = gr.Textbox(label="Predicted Arabic", lines=1)
                score_out = gr.Textbox(label="Score (vs reference)", lines=1)
                details_out = gr.Markdown()

        btn_t.click(
            fn=translit_single,
            inputs=[name_in, engine_pick, ref_in, thresh_t],
            outputs=[ar_out, score_out, details_out],
        )

        gr.Examples(
            examples=[
                ["Mohammed Ali", "model (XGBoost)", "محمد علي", 85],
                ["Ayman El Desouky", "model (XGBoost)", "أيمن الدسوقي", 85],
                ["Ahmad Hassan", "hybrid", "أحمد حسن", 90],
                ["Tariq Da'na", "model (XGBoost)", "طارق دعنا", 85],
                ["Abdennour Benantar", "rules (deterministic)", "", 85],
            ],
            inputs=[name_in, engine_pick, ref_in, thresh_t],
        )

    with gr.Tab("2. Similarity"):
        gr.Markdown("### Arabic ↔ Arabic fuzzy similarity")
        gr.Markdown(
            "Scoring is lenient — tashkeel stripped, hamza/taa-marbuta/alef-maksura unified, "
            "then `max(fuzz.ratio, fuzz.partial_ratio)` via rapidfuzz."
        )
        with gr.Row():
            with gr.Column():
                a_in = gr.Textbox(label="Arabic string A", placeholder="أحمد حسن", lines=1)
                b_in = gr.Textbox(label="Arabic string B", placeholder="احمد حسن", lines=1)
                thresh_s = gr.Slider(
                    minimum=0, maximum=100, value=85, step=1,
                    label="Pass threshold",
                )
                btn_s = gr.Button("Compare", variant="primary")

            with gr.Column():
                verdict_out = gr.Textbox(label="Result", lines=1)
                sim_score_out = gr.Textbox(label="Score", lines=1)
                sim_details_out = gr.Markdown()

        btn_s.click(
            fn=similarity_pair,
            inputs=[a_in, b_in, thresh_s],
            outputs=[verdict_out, sim_score_out, sim_details_out],
        )

        gr.Examples(
            examples=[
                ["أحمد حسن", "احمد حسن", 85],
                ["مروة فرج", "مروه فرج", 85],
                ["محمد علي", "محمد علي", 85],
                ["أدهم ساولي", "أدهم الصولي", 85],
            ],
            inputs=[a_in, b_in, thresh_s],
        )

    with gr.Tab("3. Batch"):
        gr.Markdown("### Batch transliteration")
        gr.Markdown("Paste one English name per line. Output is a markdown table + downloadable CSV.")
        with gr.Row():
            with gr.Column():
                batch_in = gr.Textbox(
                    label="English names (one per line)",
                    placeholder="Mohammed Ali\nAhmad Hassan\nMarwa Farag",
                    lines=10,
                )
                batch_engine = gr.Radio(
                    ["model (XGBoost)", "rules (deterministic)", "hybrid"],
                    value="model (XGBoost)",
                    label="Engine",
                )
                btn_b = gr.Button("Transliterate batch", variant="primary")

            with gr.Column():
                batch_md = gr.Markdown()
                batch_csv = gr.Textbox(
                    label="CSV output (copy / paste)",
                    lines=10,
                )

        btn_b.click(
            fn=batch_transliterate,
            inputs=[batch_in, batch_engine],
            outputs=[batch_md, batch_csv],
        )

    gr.Markdown(
        """
---

**About:** arabnamer is an open-source Python library extracted from an MSc-thesis
project on Arabic name handling. The model, dataset, and training code are all public
and reproducible. Built for KYC / compliance / on-premise entity resolution where
names cannot be sent to cloud APIs.

**License:** code MIT · dataset + model weights CC-BY-4.0.

Maintained by [Elsayed Yousef](mailto:elsayed.yousef@gmail.com) ·
[Commercial support available](mailto:elsayed.yousef@gmail.com).
        """
    )


if __name__ == "__main__":
    demo.launch()
