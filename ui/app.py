"""Plant disease identification UI (COS30082 ML project, Member E).

Run from the repo root:   python ui\\app.py        (or double-click ui\\run_ui.bat)
Opens http://127.0.0.1:7860 in the browser. Runs fully offline once the BioCLIP base files are cached.
"""
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from predictor import ASSETS, Predictor, SimilarIndex  # noqa: E402
from preprocessing import load_image  # noqa: E402
import views  # noqa: E402

UI = Path(__file__).resolve().parent
P = Predictor()
INDEX = SimilarIndex()
EXAMPLES = json.loads((ASSETS / 'examples.json').read_text(encoding='utf-8')) if (ASSETS / 'examples.json').is_file() else []
EXAMPLE_TRUTH = {}
for ex in EXAMPLES:
    comp = next((c for c in range(P.labels.taxonomy.size('composition')) if P.labels.pair_title(c) == ex['title']), None)
    if comp is not None:
        EXAMPLE_TRUTH[Path(ex['file']).name] = {
            'id': comp, 'title': ex['title'], 'unseen': not bool(P.labels.seen_mask[comp]),
            'crop': int(P.labels.comp_crop[comp]), 'disease': int(P.labels.comp_disease[comp])}
SEEN_COUNTS = {int(c): int(n) for c, n in zip(*np.unique(INDEX.comp, return_counts=True))} if INDEX.ok else {}


def truth_for(path):
    return EXAMPLE_TRUTH.get(Path(path).name) if path else None


def identify(path, model_key):
    if not path:
        return views.how_it_works(), ''
    try:
        image = load_image(path)
        pred = P.predict(model_key, image)
        embedding = pred['embedding'] if model_key == P.config['index_model'] else P.embed(image)
        search = INDEX.search(embedding, pred['best'])
        return (views.result_card(pred, SEEN_COUNTS.get(pred['best']['id'])),
                views.similar_card(search, pred, P.labels.pair_title))
    except Exception as exc:  # show the problem in the page instead of a silent failure
        traceback.print_exc()
        return views.error_card(str(exc)), ''


def compare(path):
    truth = truth_for(path)
    if not path:
        return views.compare_intro(None), ''
    try:
        image = load_image(path)
        order = sorted((k for k, m in P.meta.items() if m['available']), key=lambda k: P.meta[k]['group'] != 'baseline')
        preds = [P.predict(k, image) for k in order]
        return views.compare_intro(truth), views.compare_grid(preds, truth)
    except Exception as exc:
        traceback.print_exc()
        return views.compare_intro(truth), views.error_card(str(exc))


def build():
    choices = P.choices()
    default = P.config['default_model'] if any(k == P.config['default_model'] for _, k in choices) else choices[0][1]
    example_paths = [[str(ASSETS / ex['file'])] for ex in EXAMPLES]
    example_caption = 'Test images: ' + ', '.join(
        f'{ex["title"]} ({"seen" if ex["seen"] else "unseen"})' for ex in EXAMPLES) + '.'

    with gr.Blocks(title='Plant Disease ID · COS30082') as demo:
        with gr.Row(elem_id='pd-top', equal_height=True):
            gr.HTML(views.header(P.device_label), elem_id='pd-header')
            model = gr.Dropdown(choices=choices, value=default, label='Model', elem_id='pd-model', scale=0,
                                min_width=440, filterable=False)

        with gr.Tabs(elem_id='pd-tabs'):
            with gr.Tab('Identify', id='identify'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320, elem_classes='pd-left'):
                        image = gr.Image(type='filepath', label='Leaf image', sources=['upload', 'webcam', 'clipboard'],
                                         height=360, elem_id='pd-image')
                        run = gr.Button('Identify disease', variant='primary', elem_id='pd-run')
                        if example_paths:
                            gr.Examples(examples=example_paths, inputs=image, label='Try an example',
                                        examples_per_page=8)
                            gr.Markdown(example_caption,
                                        elem_classes='pd-caption')
                    with gr.Column(scale=8, min_width=480):
                        result = gr.HTML(views.how_it_works(), elem_id='pd-result')
                        similar = gr.HTML('', elem_id='pd-similar')

            with gr.Tab('Compare models', id='compare'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=280, elem_classes='pd-left'):
                        cmp_image = gr.Image(type='filepath', label='Leaf image', sources=['upload', 'webcam', 'clipboard'],
                                             height=260)
                        cmp_run = gr.Button('Run all models', variant='primary')
                        if example_paths:
                            gr.Examples(examples=example_paths, inputs=cmp_image, label='Try an example',
                                        examples_per_page=8)
                    with gr.Column(scale=9, min_width=480):
                        cmp_intro = gr.HTML(views.compare_intro(None))
                        cmp_grid = gr.HTML('')
                        gr.HTML(views.results_table(P.meta))

        # Identify: run on any new image, on the button and when the model changes.
        image.change(identify, [image, model], [result, similar], show_progress='minimal')
        run.click(identify, [image, model], [result, similar], show_progress='minimal')
        model.change(identify, [image, model], [result, similar], show_progress='minimal')
        # Keep the Compare tab on the same photo without running it until asked.
        image.change(lambda p: p, image, cmp_image, queue=False)
        cmp_run.click(compare, cmp_image, [cmp_intro, cmp_grid], show_progress='minimal')
        cmp_image.upload(compare, cmp_image, [cmp_intro, cmp_grid], show_progress='minimal')
    return demo


def theme():
    font = [gr.themes.GoogleFont('Montserrat', weights=(400, 700)), 'Arial', 'sans-serif']
    return gr.themes.Base(primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.stone,
                          font=font, radius_size=gr.themes.sizes.radius_lg).set(
        body_background_fill='#F4F5F1', body_text_color='#17201B',
        block_background_fill='#FFFFFF', block_border_color='#DDE2DA', block_radius='14px',
        button_primary_background_fill='#1F6B45', button_primary_background_fill_hover='#154D31',
        button_primary_text_color='#FFFFFF')


def main():
    print(f'Loading the default model on {P.device_label}...')
    P.load(P.config['default_model'])  # first prediction is then fast
    build().launch(theme=theme(), css=(UI / 'style.css').read_text(encoding='utf-8'),
                   inbrowser=not os.environ.get('PD_NO_BROWSER'),
                   server_name='127.0.0.1', server_port=int(os.environ.get('PORT', 7860)),
                   allowed_paths=[str(ASSETS)])


if __name__ == '__main__':
    main()
