import json
import os

import safetensors.torch
import torch

from modules import shared, sd_hijack, sd_models

try:
    from modules import sd_models_xl

    xl = True
except:
    xl = False


def prune_model(model, isxl=False):
    keys = list(model.keys())
    base_prefix = "conditioner." if isxl else "cond_stage_model."
    for k in keys:
        if "diffusion_model." not in k and "first_stage_model." not in k and base_prefix not in k:
            model.pop(k, None)
    return model


def to_half(sd):
    for key in sd.keys():
        if 'model' in key and sd[key].dtype in {torch.float32, torch.float64, torch.bfloat16}:
            sd[key] = sd[key].half()
    return sd


def savemodel(state_dict, currentmodel, file_name, savesets, metadata={}):
    if state_dict is None:
        if shared.sd_model and shared.sd_model.sd_checkpoint_info:
            metadata = shared.sd_model.sd_checkpoint_info.metadata.copy()
        else:
            return "Current model is not a valid merged model"

        checkpoint_info = shared.sd_model.sd_checkpoint_info
        # check if current merged model is a fake checkpoint_info
        if checkpoint_info is not None:
            filename = checkpoint_info.filename
            name = os.path.basename(filename)
            info = sd_models.get_closet_checkpoint_match(name)
            if info == checkpoint_info:
                # this is a valid checkpoint_info
                # no need to save
                return "Current model is not a merged model or you've already saved model"

        # prepare metadata
        save_metadata = "save metadata" in savesets
        if save_metadata:
            metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])
        else:
            metadata = {"format": "pt"}

        if shared.sd_model is not None:
            print("load from shared.sd_model..")

            # restore textencoder
            sd_hijack.model_hijack.undo_hijack(shared.sd_model)

            for name, module in shared.sd_model.named_modules():
                if hasattr(module, "network_weights_backup"):
                    module = network_restore_weights_from_backup(module)

            state_dict = shared.sd_model.state_dict()

            sd_hijack.model_hijack.hijack(shared.sd_model)
        else:
            return "No current loaded model found"

        # name_for_extra was set with the currentmodel
        currentmodel = checkpoint_info.name_for_extra

    if "fp16" in savesets:
        pre = ".fp16"
    else:
        pre = ""
    ext = ".safetensors" if "safetensors" in savesets else ".ckpt"

    # is it a inpainting or instruct-pix2pix2 model?
    if "model.diffusion_model.input_blocks.0.0.weight" in state_dict.keys():
        shape = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape
        if shape[1] == 9:
            pre += "-inpainting"
        if shape[1] == 8:
            pre += "-instruct-pix2pix"

    if not file_name or file_name == "":
        file_name = currentmodel.replace(" ", "").replace(",", "_").replace("(", "_").replace(")", "_") + pre + ext
        if file_name[0] == "_":
            file_name = file_name[1:]
    else:
        if ext in file_name:
            file_name = file_name
        else:
            file_name = file_name + pre + ext

    file_name = os.path.join(sd_models.model_path, file_name)
    file_name = file_name.replace("ProgramFiles_x86_", "Program Files (x86)")

    if len(file_name) > 255:
        file_name.replace(ext, "")
        file_name = file_name[:240] + ext

    # check if output file already exists
    if os.path.isfile(file_name) and not "overwrite" in savesets:
        _err_msg = f"Output file ({file_name}) existed and was not saved]"
        print(_err_msg)
        return _err_msg

    print("Saving...")
    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in state_dict
    if isxl:
        # prune share memory tensors, "cond_stage_model." prefixed base tensors are share memory with "conditioner." prefixed tensors
        for i, key in enumerate(state_dict.keys()):
            if "cond_stage_model." in key:
                del state_dict[key]

    if "fp16" in savesets:
        state_dict = to_half(state_dict)
    if "prune" in savesets:
        state_dict = prune_model(state_dict, isxl)

    try:
        if ext == ".safetensors":
            safetensors.torch.save_file(state_dict, file_name, metadata=metadata)
        else:
            torch.save(state_dict, file_name)
    except Exception as e:
        print(f"ERROR: Couldn't saved:{file_name},ERROR is {e}")
        return f"ERROR: Couldn't saved:{file_name},ERROR is {e}"
    print("Done!")
    return "Merged model saved in " + file_name


def filenamecutter(name, model_a=False):
    if name == "" or name == []:
        return
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    name = os.path.splitext(checkpoint_info.filename)[0]

    if not model_a:
        name = os.path.basename(name)
    return name


from typing import Union


def network_restore_weights_from_backup(self: Union[
    torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return self

    with torch.no_grad():
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight = torch.nn.Parameter(
                    weights_backup[0].detach().requires_grad_(self.in_proj_weight.requires_grad))
                self.out_proj.weight = torch.nn.Parameter(
                    weights_backup[1].detach().requires_grad_(self.out_proj.weight.requires_grad))
            else:
                self.weight = torch.nn.Parameter(weights_backup.detach().requires_grad_(self.weight.requires_grad))

        if bias_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.out_proj.bias = torch.nn.Parameter(
                    bias_backup.detach().requires_grad_(self.out_proj.bias.requires_grad))
            else:
                self.bias = torch.nn.Parameter(bias_backup.detach().requires_grad_(self.bias.requires_grad))
        else:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.out_proj.bias = None
            else:
                self.bias = None
    return self


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None
