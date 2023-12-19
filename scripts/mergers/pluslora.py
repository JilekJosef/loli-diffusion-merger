import os
import shlex
import traceback
from pathlib import Path

import gradio as gr

import scripts.kohaku.extract_locon as extract_locon
import modules.shared as shared
import scripts.mergers.components as components
from modules import sd_models
from modules.ui import create_refresh_button
from scripts.mergers.model_util import (filenamecutter)

import scripts.kohyas.svd_merge_lora
import scripts.kohyas.extract_lora_from_models
import scripts.kohaku.merge
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'extensions-builtin', 'Lora')))
import lora
sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'extensions-builtin', 'Lora')))

selectable = []


def on_ui_tabs():
    global selectable
    selectable = [x[0] for x in lora.available_loras.items()]

    with gr.Blocks(analytics_enabled=False):
        with gr.Row(equal_height=False):
            with gr.Column(equal_height=False):
                with gr.Row():
                    sml_update = gr.Button(elem_id="calcloras", value="update list", variant='primary')
                    sml_selectall = gr.Button(elem_id="sml_selectall", value="select all", variant='primary')
                    sml_deselectall = gr.Button(elem_id="slm_deselectall", value="deselect all", variant='primary')
                    hidenb = gr.Checkbox(value=False, visible=False)
                sml_loras = gr.CheckboxGroup(label="LoRAs on disk", choices=selectable, type="value", interactive=True,
                                             visible=True)
                sml_loranames = gr.Textbox(
                    label='LoRAname1:ratio1,LoRAname2:ratio2,...',
                    lines=1,
                    value="", visible=True)

                sml_selectall.click(fn=lambda x: gr.update(value=selectable), outputs=[sml_loras])
                sml_deselectall.click(fn=lambda x: gr.update(value=[]), outputs=[sml_loras])

                with gr.Row():
                    with gr.Column(equal_height=False):
                        sml_model_a = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                                  label="Checkpoint A (Base)", interactive=True)
                        create_refresh_button(sml_model_a, sd_models.list_models,
                                              lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")
                    with gr.Column(equal_height=False):
                        sml_model_b = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="model_converter_model_name",
                                                  label="Checkpoint B (Finetune)", interactive=True)
                        create_refresh_button(sml_model_b, sd_models.list_models,
                                              lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")

                with gr.Accordion("Set dimensions", open=False):
                    sml_dim = gr.Slider(label="dim", minimum=1, maximum=1280, step=1, value=32)
                    sml_conv_dim = gr.Slider(label="conv dim", minimum=1, maximum=512, step=1, value=16)
                    sml_quantile = gr.Slider(label="Lycoris quantile", minimum=0, maximum=1, step=0.0001, value=0.99)
                    sml_dim_full_auto = gr.Checkbox(label="Use LyCoris for full auto dimensions (Extract LoRA only)")

            with gr.Column(equal_height=False):
                with gr.Row(equal_height=False):
                    with gr.Column(equal_height=False):
                        with gr.Row(equal_height=False):
                            sml_filename = gr.Textbox(label="filename(option)", lines=1, visible=True, interactive=True)
                        with gr.Row(equal_height=False):
                            precision = gr.Radio(label="save precision", choices=["float", "fp16", "bf16"],
                                                 value="fp16", type="value")
                    with gr.Column(equal_height=False):
                        sml_cpmerge = gr.Button(elem_id="model_merger_merge", value="Merge to Checkpoint",
                                                variant='primary')
                        sml_merge = gr.Button(elem_id="model_merger_merge", value="Merge LoRAs", variant='primary')
                        sml_makelora = gr.Button(elem_id="model_merger_merge",
                                                 value="Extract LoRA (B-A)",
                                                 variant='primary')
                        sml_do_resize = gr.Button(elem_id="model_merger_merge", value="Resize LoRAs",
                                                  variant='primary')

                sml_submit_result = gr.Textbox(label="Message")

        components.sml_loranames = [sml_loras, sml_loranames, hidenb]

        sml_merge.click(
            fn=lmerge,
            inputs=[sml_loranames, sml_filename, sml_dim, sml_conv_dim, precision],
            outputs=[sml_submit_result]
        )

        sml_do_resize.click(
            fn=do_resize,
            inputs=[sml_loranames, sml_filename, sml_dim, sml_conv_dim, precision],
            outputs=[sml_submit_result]
        )

        sml_makelora.click(
            fn=makelora,
            inputs=[sml_model_a, sml_model_b, sml_dim, sml_conv_dim, sml_dim_full_auto, sml_quantile, sml_filename,
                    precision],
            outputs=[sml_submit_result]
        )

        sml_cpmerge.click(
            fn=cpmerge,
            inputs=[sml_loranames, sml_filename, sml_model_a, precision],
            outputs=[sml_submit_result]
        )

        llist = {}

        def updateloras():
            lora.list_available_loras()
            names = []
            dels = []
            for n in lora.available_loras.items():
                if n[0] not in llist:
                    llist[n[0]] = ""
                names.append(n[0])
            for l in list(llist.keys()):
                if l not in names:
                    llist.pop(l)

            global selectable
            selectable = [f"{x[0]}({x[1]})" for x in llist.items()]
            return gr.update(choices=[f"{x[0]}({x[1]})" for x in llist.items()])

        sml_update.click(fn=updateloras, outputs=[sml_loras])

        def llister(names, hiden):
            ratio = 1
            if hiden:
                return gr.update()
            if names == []:
                return ""
            else:
                for i, n in enumerate(names):
                    if "(" in n:
                        names[i] = n[:n.rfind("(")]
                return f":{ratio},".join(names) + f":{ratio} "

        hidenb.change(fn=lambda x: False, outputs=[hidenb])
        sml_loras.change(fn=llister, inputs=[sml_loras, hidenb], outputs=[sml_loranames])


# resize LoRA
def do_resize(loranames, filename, dim, conv_dim, precision):
    try:
        loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]
        if any([x is None for x in loras_on_disk]):
            lora.list_available_loras()

            loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]

        # LoRAname1:ratio1,LoRAname2:ratio2,.
        lnames = loranames.split(",")
        temp = []
        for n in lnames:
            temp.append(n.split(":"))
        lnames = temp

        dim = int(dim)

        out_str = ""
        for lname in lnames:
            filename = filename + "-" + lname[0]
            if ".safetensors" not in filename:
                filename += ".safetensors"
            filename = os.path.join(shared.cmd_opts.lora_dir, filename)

            parser = scripts.kohyas.svd_merge_lora.setup_parser()
            text_args = "--save_precision " + precision + " --save_to \"" + filename + "\" --models "
            text_args = text_args + "\"" + lora.available_loras.get(lname[0], None).filename + "\" "
            text_args = text_args + "--ratios 1 "
            text_args = text_args + "--new_rank " + str(dim) + " --new_conv_rank " + str(conv_dim)

            args = parser.parse_args(shlex.split(text_args))
            scripts.kohyas.svd_merge_lora.merge(args)
            out_str = out_str + "saved: " + filename + "\n"

        return out_str
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        return exc_value


# make LoRA from checkpoint
def makelora(model_a, model_b, dim, conv_dim, auto_dim, auto_quantile, filename, precision):
    print("make LoRA start")
    if model_a == "" or model_b == "":
        return "ERROR: No model Selected"

    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
    print(f"Loading {model_a}")
    theta_0 = sd_models.read_state_dict(checkpoint_info.filename)

    is_sdxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0.keys()
    is_sd2 = "cond_stage_model.model.transformer.resblocks.0.attn.out_proj.weight" in theta_0.keys()
    del theta_0
    is_sd1 = not is_sdxl and not is_sd2
    print(f"Detected model type: SDXL: {is_sdxl}, SD2.X: {is_sd2}, SD1.X: {is_sd1}")

    sd_models.unload_model_weights()

    if filename == "":
        filename = makeloraname(model_a, model_b)
    if ".safetensors" not in filename:
        filename += ".safetensors"
    filename = os.path.join(shared.cmd_opts.lora_dir, filename)

    if not auto_dim:
        text_args = ""
        if is_sdxl:
            text_args = text_args + "--sdxl "
        elif is_sd2:
            text_args = text_args + "--v2 --v_parameterization True "
        text_args = text_args + "--save_precision " + precision + " "
        text_args = text_args + "--model_org \"" + fullpathfromname(model_a) + "\" --model_tuned \"" + fullpathfromname(model_b) + "\" --save_to \"" + filename + "\" "
        text_args = text_args + "--dim " + str(dim) + " --conv_dim " + str(conv_dim)

        parser = scripts.kohyas.extract_lora_from_models.setup_parser()
        args = parser.parse_args(shlex.split(text_args))
        scripts.kohyas.extract_lora_from_models.svd(args)

    else:
        text_args = "\"" + fullpathfromname(model_a) + "\" \"" + fullpathfromname(model_b) + "\" \"" + filename + "\" "
        if is_sd2:
            text_args = text_args + "--is_v2 "
        elif is_sdxl:
            text_args = text_args + "--is_sdxl "
        text_args = text_args + "--safetensors --mode quantile --linear_quantile " + str(auto_quantile) + " --conv_quantile " + str(auto_quantile)
        extract_locon.main(text_args)

    return "saved: " + filename


# merge LoRAs
def lmerge(loranames, filename, dim, conv_dim, precision):
    try:
        loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]
        if any([x is None for x in loras_on_disk]):
            lora.list_available_loras()

            loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]

        # LoRAname1:ratio1,LoRAname2:ratio2,.
        lnames = loranames.split(",")
        temp = []
        for n in lnames:
            temp.append(n.split(":"))
        lnames = temp

        if filename == "":
            filename = loranames.replace(",", "+").replace(":", "_")
        if ".safetensors" not in filename:
            filename += ".safetensors"
        filename = os.path.join(shared.cmd_opts.lora_dir, filename)

        dim = int(dim)

        parser = scripts.kohyas.svd_merge_lora.setup_parser()
        text_args = "--save_precision " + precision + " --save_to \"" + filename + "\" --models "
        for lname in lnames:
            text_args = text_args + "\"" + lora.available_loras.get(lname[0], None).filename + "\" "
        text_args = text_args + "--ratios "
        for lname in lnames:
            text_args = text_args + lname[1] + " "
        text_args = text_args + "--new_rank " + str(dim) + " --new_conv_rank " + str(conv_dim)

        args = parser.parse_args(shlex.split(text_args))
        scripts.kohyas.svd_merge_lora.merge(args)

        return "saved: " + filename
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        return exc_value


# merge to checkpoint
def cpmerge(loranames, filename, model, precision):
    if model == []:
        return "ERROR: No model Selected"
    if loranames == "":
        return "ERROR: No LoRA Selected"

    lnames = loranames.split(",")
    temp = []
    for n in lnames:
        temp.append(n.split(":"))
    lnames = temp

    names, filenames, loratypes, lweis = [], [], [], []

    modeln = filenamecutter(model, True)
    dname = modeln
    for n in names:
        dname = dname + "+" + n

    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    print(f"Loading {model}")
    theta_0 = sd_models.read_state_dict(checkpoint_info.filename, "cpu")

    is_sdxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0.keys()
    is_sd2 = "cond_stage_model.model.transformer.resblocks.0.attn.out_proj.weight" in theta_0.keys()
    del theta_0

    out_text = ""
    for lname in lnames:
        filename = filename + "-" + lname[0]
        if ".safetensors" not in filename:
            filename += ".safetensors"
        base_path = ""
        if shared.cmd_opts.ckpt_dir is None:
            base_path = os.path.join(Path(shared.cmd_opts.lora_dir).parent.absolute(), "Stable-diffusion")
        else:
            base_path = shared.cmd_opts.ckpt_dir
        filename = os.path.join(base_path, filename)

        text_args = "\"" + fullpathfromname(model) + "\" \"" + lora.available_loras.get(lname[0], None).filename + "\" \"" + filename + "\" "
        if is_sd2:
            text_args = text_args + "--is_v2 "
        elif is_sdxl:
            text_args = text_args + "--is_sdxl "
        text_args = text_args + "--weight " + str(lname[1]) + " "
        text_args = text_args + "--dtype " + precision
        scripts.kohaku.merge.main(text_args)
        out_text = out_text + "saved: " + filename + "\""

    return out_text


def fullpathfromname(name):
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    return checkpoint_info.filename


def makeloraname(model_a, model_b):
    model_a = filenamecutter(model_a)
    model_b = filenamecutter(model_b)
    return "lora_" + model_a + "-" + model_b
