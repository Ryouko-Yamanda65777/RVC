import logging
import os

import gradio as gr
from dotenv import load_dotenv

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.modules.vc.modules import VC

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

i18n = I18nAuto()
logger.info(i18n)

load_dotenv()
config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root")
index_root = os.getenv("index_root")
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))

# Simplified UI with organized sections
app = gr.Blocks()

with app:
    with gr.Tabs():
        gr.Markdown("#RVC RYOUKO GUI")
        with gr.TabItem("Inference"):
            with gr.Box():
                sid = gr.Dropdown(label=i18n("Select Speaker"), choices=sorted(names))
                vc_input3 = gr.Audio(label="Upload Audio (less than 90 seconds)")
                vc_transform0 = gr.Number(label="Pitch Shift (semitones)", value=0)
                but0 = gr.Button("Convert", variant="primary")

            with gr.Accordion("Advanced Options", open=False):
                f0method0 = gr.Radio(
                    label="Pitch Extraction Algorithm",
                    choices=["pm", "harvest", "crepe", "rmvpe"],
                    value="rmvpe"
                )
                filter_radius0 = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label="Median Filter Radius (for harvest algorithm)",
                    value=3,
                    step=1
                )
                file_index2 = gr.Dropdown(
                    label="Select Feature Search Library",
                    choices=sorted(index_paths),
                    interactive=True
                )
                index_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Feature Search Proportion",
                    value=0.88
                )
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label="Resample to Final Sampling Rate (0 for no resampling)",
                    value=0,
                    step=1
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Source to Output Envelope Mix Ratio",
                    value=1
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect Consonants and Breaths (set to 0.5 to disable)",
                    value=0.33,
                    step=0.01
                )
                f0_file = gr.File(label="F0 Curve File (Optional)")

            # Output section
            vc_output1 = gr.Textbox(label="Output Information")
            vc_output2 = gr.Audio(label="Output Audio (Download using bottom-right dots)")

            # Action on button click
            but0.click(
                vc.vc_single,
                [
                    sid,
                    vc_input3,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index2,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2]
            )

app.launch(share=True)
