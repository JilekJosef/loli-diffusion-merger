import gc
import os
import os.path
import re
from tqdm import tqdm
import torch
from statistics import mean
import torch.nn as nn
import torch.nn.functional as F
from importlib import reload
import gradio as gr
from modules import (script_callbacks, sd_models, sd_vae, shared)
from modules.scripts import basedir
from modules.ui import create_refresh_button
import scripts.mergers.mergers
import scripts.mergers.components as components
from importlib import reload

reload(scripts.mergers.mergers)
import csv
import scripts.mergers.pluslora as pluslora
from scripts.mergers.mergers import (EXCLUDE_CHOICES, rwmergelog, blockfromkey)

from scripts.mergers.model_util import filenamecutter

from scripts.mergers.mergers import smergegen

path_root = basedir()

CALCMODES = ["normal", "cosineA", "cosineB", "trainDifference", "smoothAdd", "smoothAdd MT", "extract", "tensor",
             "tensor2", "self"]


class ResizeHandleRow(gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.elem_classes.append("resize-handle-row")

    def get_block_name(self):
        return "row"


from typing import Union


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None


def fix_network_reset_cached_weight():
    try:
        import networks as net
        net.network_reset_cached_weight = network_reset_cached_weight
    except:
        pass


weights_presets = ""
userfilepath = os.path.join(path_root, "scripts", "mbwpresets.txt")
if os.path.isfile(userfilepath):
    try:
        with open(userfilepath) as f:
            weights_presets = f.read()
    except OSError as e:
        print(e)
        pass

def on_ui_tabs():
    fix_network_reset_cached_weight()

    with gr.Blocks() as loli_diffusion_merger_ui:
        with gr.Tab("Merge"):
            with ResizeHandleRow(equal_height=False):
                with gr.Column(variant="compact"):
                    with gr.Row(variant="compact"):
                        model_a = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                              label="Model A", interactive=True)
                        create_refresh_button(model_a, sd_models.list_models,
                                              lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")

                        model_b = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                              label="Model B", interactive=True)
                        create_refresh_button(model_b, sd_models.list_models,
                                              lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")

                        model_c = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                              label="Model C", interactive=True)
                        create_refresh_button(model_c, sd_models.list_models,
                                              lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")

                    mode = gr.Radio(label="Merge Mode",
                                    choices=["Weight sum", "Add difference", "Triple sum", "Sum twice", "AND gate"],
                                    value="Weight sum", info="A*(1-alpha)+B*alpha Supported calculation methods: normal, cosineA (prioritize structure of A), cosineB (prioritize structure of B), tensor (replace some weights in ratio instead of summing), tensor2 (when the tensor has a large number of dimensions, exchanges are performed based on the second dimension), self (ignores second model and multiply model A with alpha -> will likely result in model corruption if not used carefully)")
                    calcmode = gr.Radio(label="Calculation Mode", choices=CALCMODES, value="normal")
                    with gr.Row(variant="compact"):
                        with gr.Column(scale=1):
                            useblocks = gr.Checkbox(label="use MBW", info="use Merge Block Weights")
                        with gr.Column(scale=3), gr.Group() as alpha_group:
                            with gr.Row():
                                base_alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                                base_beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=0.25,
                                                      interactive=False)
                            with gr.Row():
                                base_deviation = gr.Slider(label="deviation", minimum=0, maximum=100, step=0.0000001,
                                                           value=5)
                        # weights = gr.Textbox(label="weights,base alpha,IN00,IN02,...IN11,M00,OUT00,...,OUT11",lines=2,value="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")

                    with gr.Accordion("Options", open=True):
                        with gr.Row(variant="compact"):
                            save_sets = gr.CheckboxGroup(
                                ["overwrite", "safetensors", "fp16", "save metadata", "prune",
                                 "Reset CLIP ids"], value=["safetensors"],
                                show_label=False, label="save settings")
                            opt_value = gr.Slider(label="option", minimum=-1.0, maximum=2, step=0.001, value=0.3,
                                                  interactive=True)
                        with gr.Row(variant="compact"):
                            with gr.Column(min_width=50):
                                with gr.Row():
                                    custom_name = gr.Textbox(label="Custom Name (Optional)",
                                                             elem_id="model_converter_custom_name")
                            with gr.Column():
                                with gr.Row():
                                    bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None",
                                                              label="Bake in VAE", elem_id="modelmerger_bake_in_vae")
                                    create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list,
                                                          lambda: {"choices": ["None"] + list(sd_vae.vae_dict)},
                                                          "modelmerger_refresh_bake_in_vae")

                    with gr.Accordion("Merging Block Weights", open=False):
                        with gr.Row():
                            isxl = gr.Radio(label="Block Type", choices=["1.X or 2.X", "XL"], value="1.X or 2.X",
                                            type="index")

                        with gr.Tab("Weights Setting"):
                            with gr.Group(), gr.Tabs():
                                with gr.Tab("Weights for alpha"):
                                    with gr.Row(variant="compact"):
                                        weights_a = gr.Textbox(label="BASE,IN00,IN02,...IN11,M00,OUT00,...,OUT11",
                                                               value="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5",
                                                               show_copy_button=True)
                                    with gr.Row(scale=2):
                                        setalpha = gr.Button(elem_id="copytogen", value="↑ Set alpha",
                                                             variant='primary', scale=3)
                                        readalpha = gr.Button(elem_id="copytogen", value="↓ Read alpha",
                                                              variant='primary', scale=3)
                                        setx = gr.Button(elem_id="copytogen", value="↑ Set X", min_width="80px",
                                                         scale=1)
                                with gr.Tab("beta"):
                                    with gr.Row(variant="compact"):
                                        weights_b = gr.Textbox(label="BASE,IN00,IN02,...IN11,M00,OUT00,...,OUT11",
                                                               value="0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2",
                                                               show_copy_button=True)
                                    with gr.Row(scale=2):
                                        setbeta = gr.Button(elem_id="copytogen", value="↑ Set beta", variant='primary',
                                                            scale=3)
                                        readbeta = gr.Button(elem_id="copytogen", value="↓ Read beta",
                                                             variant='primary', scale=3)
                                        sety = gr.Button(elem_id="copytogen", value="↑ Set Y", min_width=80,
                                                         scale=1)

                            with gr.Group(), gr.Tabs():
                                with gr.Tab("Preset"):
                                    with gr.Row():
                                        dd_preset_weight = gr.Dropdown(label="Select preset",
                                                                       choices=preset_name_list(weights_presets),
                                                                       interactive=True, elem_id="refresh_presets")

                            with gr.Row():
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)
                                with gr.Column(scale=2, min_width=200):
                                    base = gr.Slider(label="Base", minimum=0, maximum=1, step=0.0001, value=0.5)
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)
                            with gr.Row():
                                with gr.Column(scale=2, min_width=200):
                                    in00 = gr.Slider(label="IN00", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in01 = gr.Slider(label="IN01", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in02 = gr.Slider(label="IN02", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in03 = gr.Slider(label="IN03", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in04 = gr.Slider(label="IN04", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in05 = gr.Slider(label="IN05", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in06 = gr.Slider(label="IN06", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in07 = gr.Slider(label="IN07", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in08 = gr.Slider(label="IN08", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in09 = gr.Slider(label="IN09", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in10 = gr.Slider(label="IN10", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in11 = gr.Slider(label="IN11", minimum=0, maximum=1, step=0.0001, value=0.5)
                                with gr.Column(scale=2, min_width=200):
                                    ou11 = gr.Slider(label="OUT11", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou10 = gr.Slider(label="OUT10", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou09 = gr.Slider(label="OUT09", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou08 = gr.Slider(label="OUT08", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou07 = gr.Slider(label="OUT07", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou06 = gr.Slider(label="OUT06", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou05 = gr.Slider(label="OUT05", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou04 = gr.Slider(label="OUT04", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou03 = gr.Slider(label="OUT03", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou02 = gr.Slider(label="OUT02", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou01 = gr.Slider(label="OUT01", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou00 = gr.Slider(label="OUT00", minimum=0, maximum=1, step=0.0001, value=0.5)
                            with gr.Row():
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)
                                with gr.Column(scale=2, min_width=200):
                                    mi00 = gr.Slider(label="M00", minimum=0, maximum=1, step=0.0001, value=0.5)
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)

                    components.dtrue = gr.Checkbox(value=True, visible=False)
                    components.dfalse = gr.Checkbox(value=False, visible=False)
                    dummy_t = gr.Textbox(value="", visible=False)

                with gr.Column(variant="compact"):
                    with gr.Row():
                        components.merge = gr.Button(elem_id="model_merger_merge", elem_classes=["compact_button"],
                                                     value="Merge!", variant='primary')
                    components.currentmodel = gr.Textbox(label="Current Model", lines=1, value="", interactive=False)
                    components.submit_result = gr.Textbox(label="Message", interactive=False)

        # main ui end
        with gr.Tab("LoRA", elem_id="tab_lora"):
            pluslora.on_ui_tabs()

        with gr.Tab("Analysis", elem_id="tab_analysis"):
            with gr.Tab("Models"):
                with gr.Row():
                    an_model_a = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                             label="Checkpoint A", interactive=True)
                    create_refresh_button(an_model_a, sd_models.list_models,
                                          lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")
                    an_model_b = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                             label="Checkpoint B", interactive=True)
                    create_refresh_button(an_model_b, sd_models.list_models,
                                          lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")
                with gr.Row():
                    an_mode = gr.Radio(label="Analysis Mode", choices=["ASimilarity", "Block", "Element", "Both"],
                                       value="ASimilarity", type="value")
                    an_calc = gr.Radio(label="Block method", choices=["Mean", "Min", "attn2"], value="Mean",
                                       type="value")
                    an_include = gr.CheckboxGroup(label="Include", choices=["Textencoder(BASE)", "U-Net", "VAE"],
                                                  value=["Textencoder(BASE)", "U-Net"], type="value")
                    an_settings = gr.CheckboxGroup(label="Settings", choices=["save as txt", "save as csv"],
                                                   type="value", interactive=True)
                with gr.Row():
                    run_analysis = gr.Button(value="Run Analysis", variant='primary')
                with gr.Row():
                    analysis_cosdif = gr.Dataframe(headers=["block", "key", "similarity[%]"], )
            with gr.Tab("Text Encoder"):
                with gr.Row():
                    te_smd_loadkeys = gr.Button(value="Calculate Textencoer", variant='primary')
                    te_smd_searchkeys = gr.Button(value="Search Word(red,blue,girl,...)", variant='primary')
                    exclude = gr.Checkbox(label="exclude non numeric,alphabet,symbol word")
                pickupword = gr.TextArea()
                encoded = gr.Dataframe()

        run_analysis.click(fn=calccosinedif, inputs=[an_model_a, an_model_b, an_mode, an_settings, an_include, an_calc],
                           outputs=[analysis_cosdif])

        import sys
        sys.path.insert(0, os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'extensions-builtin', 'Lora')))
        import lora
        sys.path.remove(os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'extensions-builtin', 'Lora')))

        with gr.Tab("Elements", elem_id="tab_deep"):
            with gr.Row():
                smd_model_a = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                          label="Checkpoint", interactive=True)
                create_refresh_button(smd_model_a, sd_models.list_models,
                                      lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")
                smd_loadkeys = gr.Button(value="load keys", variant='primary')
            with gr.Row():
                smd_lora = gr.Dropdown(list(lora.available_loras.keys()), elem_id="model_converter_model_name",
                                       label="LoRA", interactive=True)
                create_refresh_button(smd_lora, lora.list_available_loras,
                                      lambda: {"choices": list(lora.available_loras.keys())}, "refresh_checkpoint_Z")
                smd_loadkeys_l = gr.Button(value="load keys", variant='primary')
            with gr.Row():
                keys = gr.Dataframe(headers=["No.", "block", "key"], )

        with gr.Tab("Metadeta", elem_id="tab_metadata"):
            with gr.Row():
                meta_model_a = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                           label="read metadata", interactive=True)
                create_refresh_button(meta_model_a, sd_models.list_models,
                                      lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")
                smd_loadmetadata = gr.Button(value="load keys", variant='primary')
            with gr.Row():
                metadata = gr.TextArea()

        smd_loadmetadata.click(
            fn=loadmetadata,
            inputs=[meta_model_a],
            outputs=[metadata]
        )

        smd_loadkeys.click(fn=loadkeys, inputs=[smd_model_a, components.dfalse], outputs=[keys])
        smd_loadkeys_l.click(fn=loadkeys, inputs=[smd_lora, components.dtrue], outputs=[keys])

        te_smd_loadkeys.click(fn=encodetexts, inputs=[exclude], outputs=[encoded])
        te_smd_searchkeys.click(fn=pickupencode, inputs=[pickupword], outputs=[encoded])

        components.msettings = [weights_a, weights_b, model_a, model_b, model_c, base_alpha, base_beta, base_deviation,
                                mode, calcmode,
                                useblocks, custom_name, save_sets, bake_in_vae, opt_value]

        components.merge.click(
            fn=smergegen,
            inputs=[*components.msettings],
            outputs=[components.submit_result, components.currentmodel]
        )

        menbers = [base, in00, in01, in02, in03, in04, in05, in06, in07, in08, in09, in10, in11, mi00, ou00, ou01, ou02,
                   ou03, ou04, ou05, ou06, ou07, ou08, ou09, ou10, ou11]

        setalpha.click(fn=slider2text, inputs=[*menbers, dd_preset_weight, isxl], outputs=[weights_a])
        setbeta.click(fn=slider2text, inputs=[*menbers, dd_preset_weight, isxl], outputs=[weights_b])
        setx.click(fn=add_to_seq, inputs=[weights_a])
        sety.click(fn=add_to_seq, inputs=[weights_b])

        mode_info = {
            "Weight sum": "A*(1-alpha)+B*alpha Supported calculation methods: normal, cosineA (prioritize structure of A), cosineB (prioritize structure of B), tensor (replace some weights in ratio instead of summing), tensor2 (when the tensor has a large number of dimensions, exchanges are performed based on the second dimension), self (ignores second model and multiply model A with alpha -> will likely result in model corruption if not used carefully)",
            "Add difference": "A+(B-C)*alpha Supported calculation methods: normal, train difference (LoRA like), smooth add (uses median and gausion filter and tries to reduce noise), smooth add MT (multi-threaded smooth add -> runs faster), extract (merge common and uncommon parts between models B and C)",
            "Triple sum": "A*(1-alpha-beta)+B*alpha+C*beta Supported calculation methods: normal",
            "Sum twice": "(A*(1-alpha)+B*alpha)*(1-beta)+C*beta Supported calculation methods: normal",
            "AND gate": "A+(B AND(deviation) C) In case B and C value is similar enough, value of B is used, otherwise value of A is used. Supported calculation methods: normal"
        }
        mode.change(fn=lambda mode, calcmode: [gr.update(info=mode_info[mode]), gr.update(
            interactive=True if mode in ["Triple sum", "sum Twice"] or calcmode in ["tensor", "tensor2"] else False)],
                    inputs=[mode, calcmode], outputs=[mode, base_beta], show_progress=False)
        calcmode.change(fn=lambda calcmode: gr.update(interactive=True) if calcmode in ["tensor", "tensor2",
                                                                                        "extract"] else gr.update(),
                        inputs=[calcmode], outputs=base_beta, show_progress=False)
        useblocks.change(fn=lambda mbw: gr.update(visible=False if mbw else True), inputs=[useblocks],
                         outputs=[alpha_group])

        readalpha.click(fn=text2slider, inputs=[weights_a, isxl], outputs=menbers)
        readbeta.click(fn=text2slider, inputs=[weights_b, isxl], outputs=menbers)

        dd_preset_weight.change(fn=on_change_dd_preset_weight, inputs=[dd_preset_weight], outputs=menbers)

        def changexl(isxl):
            out = [True] * 26
            if isxl:
                for i, id in enumerate(BLOCKID[:-1]):
                    if id not in BLOCKIDXLL[:-1]:
                        out[i] = False
            return [gr.update(visible=x) for x in out]

        isxl.change(fn=changexl, inputs=[isxl], outputs=menbers)

    return (loli_diffusion_merger_ui, "Loli Diffusion Merger", "loli-diffusion-merger"),

def loadmetadata(model):
    import json
    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    if ".safetensors" not in checkpoint_info.filename: return "no metadata(not safetensors)"
    sdict = sd_models.read_metadata_from_safetensors(checkpoint_info.filename)
    if sdict == {}:
        return "no metadata"
    return json.dumps(sdict, indent=4)

def add_to_seq(seq, maker):
    return gr.Textbox.update(value=maker if seq == "" else seq + "\r\n" + maker)


def text2slider(text, isxl=False):
    vals = [t.strip() for t in text.split(",")]

    if isxl:
        j = 0
        ret = []
        for i, v in enumerate(ISXLBLOCK):
            if v:
                ret.append(gr.update(value=float(vals[j])))
                j += 1
            else:
                ret.append(gr.update())
        return ret

    return [gr.update(value=float(v)) for v in vals]


def slider2text(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, preset, isxl):
    numbers = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]
    if isxl:
        newnums = []
        for i, id in enumerate(BLOCKID[:-1]):
            if id in BLOCKIDXLL[:-1]:
                newnums.append(numbers[i])
        numbers = newnums
    numbers = [str(x) for x in numbers]
    return gr.update(value=",".join(numbers))


def on_change_dd_preset_weight(preset):
    weights = find_preset_by_name(preset)
    if weights is not None:
        return text2slider(weights)


def tagdicter(presets):
    presets = presets.splitlines()
    preset_names = []
    for line in presets:
        preset_names.append(line.split("\t")[0].strip())

    return ",".join(preset_names)


def preset_name_list(presets):
    return tagdicter(presets).split(",")


def find_preset_by_name(preset):
    presets = weights_presets.splitlines()
    for line in presets:
        if preset in line:
            return line.split("\t")[1].strip()

    return None


BLOCKID = ["BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08", "IN09", "IN10", "IN11",
           "M00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08", "OUT09", "OUT10",
           "OUT11", "Not Merge"]
BLOCKIDXL = ['BASE', 'IN0', 'IN1', 'IN2', 'IN3', 'IN4', 'IN5', 'IN6', 'IN7', 'IN8', 'M', 'OUT0', 'OUT1', 'OUT2', 'OUT3',
             'OUT4', 'OUT5', 'OUT6', 'OUT7', 'OUT8', 'VAE']
BLOCKIDXLL = ['BASE', 'IN00', 'IN01', 'IN02', 'IN03', 'IN04', 'IN05', 'IN06', 'IN07', 'IN08', 'M00', 'OUT00', 'OUT01',
              'OUT02', 'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'VAE']
ISXLBLOCK = [True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True,
             True, True, True, True, True, True, False, False, False]


def modeltype(sd):
    if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in sd.keys():
        model_type = "XL"
    else:
        model_type = "1.X or 2.X"
    return model_type


def loadkeys(model_a, lora):
    if lora:
        import Lora.lora as lora
        sd = sd_models.read_state_dict(lora.available_loras[model_a].filename, "cpu")
    else:
        sd = loadmodel(model_a)
    keys = []
    mtype = modeltype(sd)
    if lora:
        for i, key in enumerate(sd.keys()):
            keys.append([i, "LoRA", key, sd[key].shape])
    else:
        for i, key in enumerate(sd.keys()):
            keys.append([i, blockfromkey(key, mtype), key, sd[key].shape])

    return keys


def loadmodel(model):
    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    sd = sd_models.read_state_dict(checkpoint_info.filename, "cpu")
    return sd


def calccosinedif(model_a, model_b, mode, settings, include, calc):
    inc = " ".join(include)
    settings = " ".join(settings)
    a, b = loadmodel(model_a), loadmodel(model_b)
    name = filenamecutter(model_a) + "-" + filenamecutter(model_b)
    cosine_similarities = []
    blocksim = {}
    blockvals = []
    attn2 = {}
    isxl = "XL" == modeltype(a)
    blockids = BLOCKIDXLL if isxl else BLOCKID
    for bl in blockids:
        blocksim[bl] = []
    blocksim["VAE"] = []

    if "ASim" in mode:
        result = asimilarity(a, b, isxl)
        if len(settings) > 1:
            savecalc(result, name, settings, True, "Asim")
        del a, b
        gc.collect()
        return result
    else:
        for key in tqdm(a.keys(), desc="Calculating cosine similarity"):
            block = None
            if blockfromkey(key, isxl) == "Not Merge":
                continue
            if "model_ema" in key:
                continue
            if "model" not in key:
                continue
            if "first_stage_model" in key and not ("VAE" in inc):
                continue
            elif "first_stage_model" in key and "VAE" in inc:
                block = "VAE"
            if "diffusion_model" in key and not ("U-Net" in inc):
                continue
            if "encoder" in key and not ("encoder" in inc):
                continue
            if key in b and a[key].size() == b[key].size():
                a_flat = a[key].view(-1).to(torch.float32)
                b_flat = b[key].view(-1).to(torch.float32)
                simab = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))
                if block is None:
                    block, blocks26 = blockfromkey(key, isxl)
                if block == "Not Merge":
                    continue
                cosine_similarities.append([block, key, round(simab.item() * 100, 3)])
                blocksim[blocks26].append(round(simab.item() * 100, 3))
                if "attn2.to_out.0.weight" in key:
                    attn2[block] = round(simab.item() * 100, 3)

        for bl in blockids:
            val = None
            if bl == "Not Merge":
                continue
            if bl not in blocksim.keys():
                continue
            if blocksim[bl] == []:
                continue
            if "Mean" in calc:
                val = mean(blocksim[bl])
            elif "Min" in calc:
                val = min(blocksim[bl])
            else:
                if bl in attn2.keys():
                    val = attn2[bl]
            if val:
                blockvals.append([bl, "", round(val, 3)])
            if mode != "Element":
                cosine_similarities.insert(0, [bl, "", round(mean(blocksim[bl]), 3)])

        if mode == "Block":
            if len(settings) > 1:
                savecalc(blockvals, name, settings, True, "Blocks")
            del a, b
            gc.collect()
            return blockvals
        else:
            if len(settings) > 1:
                savecalc(cosine_similarities, name, settings, False, "Elements", )
            del a, b
            gc.collect()
            return cosine_similarities


def savecalc(data, name, settings, blocks, add):
    name = name + "_" + add
    csvpath = os.path.join(path_root, f"{name}.csv")
    txtpath = os.path.join(path_root, f"{name}.txt")

    txt = ""
    for row in data:
        row = [str(r) for r in row]
        txt = txt + ",".join(row) + "\n"
        if blocks:
            txt = txt.replace(",,", ",")

    if "txt" in settings:
        with open(txtpath, 'w+') as f:
            f.writelines(txt)
            print("file saved to ", txtpath)
    if "csv" in settings:
        with open(csvpath, 'w+') as f:
            f.writelines(txt)
            print("file saved to ", csvpath)


# code from https://huggingface.co/JosephusCheung/ASimilarityCalculatior

def cal_cross_attn(to_q, to_k, to_v, rand_input):
    hidden_dim, embed_dim = to_q.shape
    attn_to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_q.load_state_dict({"weight": to_q})
    attn_to_k.load_state_dict({"weight": to_k})
    attn_to_v.load_state_dict({"weight": to_v})

    return torch.einsum(
        "ik, jk -> ik",
        F.softmax(torch.einsum("ij, kj -> ik", attn_to_q(rand_input), attn_to_k(rand_input)), dim=-1),
        attn_to_v(rand_input)
    )


def eval(model, n, input, block):
    qk = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_q.weight"
    uk = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_k.weight"
    vk = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_v.weight"
    atoq, atok, atov = model[qk], model[uk], model[vk]

    attn = cal_cross_attn(atoq, atok, atov, input)
    return attn


ATTN1BLOCKS = [[1, "input"], [2, "input"], [4, "input"], [5, "input"], [7, "input"], [8, "input"], ["", "middle"],
               [3, "output"], [4, "output"], [5, "output"], [6, "output"], [7, "output"], [8, "output"], [9, "output"],
               [10, "output"], [11, "output"]]


def asimilarity(model_a, model_b, mtype):
    torch.manual_seed(2244096)
    sims = []

    for nblock in tqdm(ATTN1BLOCKS, desc="Calculating cosine similarity"):
        n, block = nblock[0], nblock[1]
        if n != "": n = f"s.{n}"
        key = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_q.weight"

        hidden_dim, embed_dim = model_a[key].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        attn_a = eval(model_a, n, rand_input, block)
        attn_b = eval(model_b, n, rand_input, block)

        sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
        sims.append([blockfromkey(key, mtype), "", round(sim.item() * 100, 3)])

    return sims


CONFIGS = ["prompt", "neg_prompt", "Steps", "Sampling method", "CFG scale", "Seed", "Width", "Height", "Batch size",
           "Upscaler", "Hires steps", "Denoising strength", "Upscale by"]
RESETVALS = ["", "", 0, " ", 0, 0, 0, 0, 1, "Latent", 0, 0.7, 2]

sorted_output = []


def encodetexts(exclude):
    isxl = hasattr(shared.sd_model, "conditioner")
    model = shared.sd_model.conditioner.embedders[0] if isxl else shared.sd_model.cond_stage_model
    encoder = model.encode_with_transformers
    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    byte_decoder = tokenizer.byte_decoder

    batch = 500

    b_texts = [list(vocab.items())[i:i + batch] for i in range(0, len(vocab), batch)]

    output = []

    for texts in tqdm(b_texts):
        batch = []
        words = []
        for word, idx in texts:
            tokens = [model.id_start, idx, model.id_end] + [model.id_end] * 74
            batch.append(tokens)
            words.append((idx, word))

        embedding = encoder(torch.IntTensor(batch).to("cuda"))[:, 1, :]  # (bs,768)
        embedding = embedding.to('cuda')
        emb_norms = torch.linalg.vector_norm(embedding, dim=-1)  # (bs,)

        for i, (word, token) in enumerate(texts):
            try:
                word = bytearray([byte_decoder[x] for x in word]).decode("utf-8")
            except UnicodeDecodeError:
                pass
            if exclude:
                if has_alphanumeric(word): output.append([word, token, emb_norms[i].item()])
            else:
                output.append([word, token, emb_norms[i].item()])

    output = sorted(output, key=lambda x: x[2], reverse=True)
    for i in range(len(output)):
        output[i].insert(0, i)

    global sorted_output
    sorted_output = output

    return output[:1000]


def pickupencode(texts):
    wordlist = [x[1] for x in sorted_output]
    texts = texts.split(",")
    output = []
    for text in texts:
        if text in wordlist:
            output.append(sorted_output[wordlist.index(text)])
        if text + "</w>" in wordlist:
            output.append(sorted_output[wordlist.index(text + "</w>")])
    return output


def has_alphanumeric(text):
    pattern = re.compile(r'[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;"\'<>,.?/\|\\]')
    return bool(pattern.search(text.replace("</w>", "")))


if __package__ == "loli-diffusion-merger":
    script_callbacks.on_ui_tabs(on_ui_tabs)
