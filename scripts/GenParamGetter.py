import gradio as gr
import scripts.mergers.components as components
from scripts.mergers.mergers import smergegen
from modules import scripts, script_callbacks


class GenParamGetter(scripts.Script):
    events_assigned = False

    def title(self):
        return "Loli Diffusion Marger Parameter Getter"

    def get_params_components(demo: gr.Blocks, app):
        if not GenParamGetter.events_assigned:
            with demo:
                components.merge.click(
                    fn=smergegen,
                    inputs=[*components.msettings],
                    outputs=[components.submit_result,components.currentmodel]
                )

            GenParamGetter.events_assigned = True


if __package__ == "GenParamGetter":
    script_callbacks.on_app_started(GenParamGetter.get_params_components)
