import gradio as gr
import spaces

from mythograph.config import APP_TITLE, MODEL_UI_DIRECTOR_ENABLED, ROOT_DIR
from mythograph.models.image_client import ImageClient
from mythograph.models.llm_client import runtime_status
from mythograph.schemas.profile import InterviewProfile
from mythograph.schemas.ui import UIAction
from mythograph.services.art_recipe import build_art_recipe_with_model
from mythograph.services.interview import (
    apply_answer,
    choose_next_ui,
    choose_next_ui_with_model,
    new_profile,
    start_with_surprise,
)
from mythograph.services.trace_logger import export_trace, log_event
from mythograph.ui.examples import REGENERATION_OPTIONS, STARTER_IDEAS


def build_demo() -> gr.Blocks:
    css = (ROOT_DIR / "mythograph" / "ui" / "styles.css").read_text(encoding="utf-8")

    with gr.Blocks(title=APP_TITLE, css=css, elem_id="ma-shell", theme=gr.themes.Base()) as demo:
        profile_state = gr.State(new_profile().model_dump())
        action_state = gr.State(UIAction.ASK_FREE_TEXT.value)
        recipe_state = gr.State(None)

        gr.HTML(
            """
            <section id="ma-hero">
              <h1>Mythograph Atelier</h1>
              <p>Create abstract art backwards: choose the meaning first, let the atelier turn it into symbols, color, and a story you can tell a friend.</p>
            </section>
            """,
            container=False,
        )

        with gr.Row(elem_classes=["ma-stage"]):
            with gr.Column(scale=4):
                assistant_message = gr.Markdown(
                    "Choose a starting idea, write your own, or ask the atelier to surprise you.",
                    elem_classes=["ma-note"],
                )
                gr.Markdown(_format_runtime_status(), elem_classes=["ma-note"])
                custom_idea = gr.Textbox(
                    label="Start with your own idea",
                    placeholder="I want something about discipline and hope.",
                    lines=2,
                )
                with gr.Row():
                    submit_text = gr.Button("Use this idea", variant="primary")
                    surprise = gr.Button("Surprise me")
                    create_now = gr.Button("Create now")

                starter = gr.Radio(
                    choices=[f"{title}: {text}" for title, text in STARTER_IDEAS],
                    label="Starting points",
                    value=None,
                )

                idea_cards = gr.CheckboxGroup(label="Idea cards", choices=[], visible=False)
                style_cards = gr.Radio(label="Style cards", choices=[], visible=False)
                symbol_cards = gr.Radio(label="Symbol cards", choices=[], visible=False)
                contrast_cards = gr.Radio(label="Contrast", choices=[], visible=False)
                visual_group = gr.Group(visible=False)
                with visual_group:
                    minimal_rich = gr.Slider(0, 100, value=35, step=1, label="minimal to rich")
                    calm_intense = gr.Slider(0, 100, value=45, step=1, label="calm to intense")
                    geometric_organic = gr.Slider(0, 100, value=35, step=1, label="geometric to organic")
                    visual_submit = gr.Button("Save visual direction", variant="primary")

                with gr.Row():
                    continue_button = gr.Button("Continue", variant="primary")
                    reset_button = gr.Button("Reset")

                debug_reason = gr.Markdown("", visible=False)

            with gr.Column(scale=5):
                image_output = gr.Image(label="Painting", type="filepath", height=620)
                gallery_label = gr.Markdown("", elem_classes=["ma-gallery-label"])
                symbol_map = gr.Markdown("", elem_classes=["ma-symbol-table"])
                regen = gr.Dropdown(
                    choices=REGENERATION_OPTIONS,
                    label="Regenerate",
                    value=None,
                    interactive=True,
                    visible=False,
                )
                regen_button = gr.Button("Apply regeneration", visible=False)
                image_prompt = gr.Textbox(label="Image prompt", lines=4, visible=False)
                with gr.Accordion("Model activity", open=False):
                    model_activity = gr.Textbox(
                        label="Activity",
                        lines=7,
                        visible=False,
                        interactive=False,
                    )
                trace_file = gr.File(label="Trace export", visible=False)
                trace_button = gr.Button("Download trace", visible=False)

        submit_text.click(
            fn=_submit_free_text,
            inputs=[profile_state, custom_idea],
            outputs=_flow_outputs(
                profile_state,
                action_state,
                assistant_message,
                idea_cards,
                style_cards,
                symbol_cards,
                contrast_cards,
                visual_group,
                debug_reason,
            ),
        )
        starter.change(
            fn=_submit_starter,
            inputs=[profile_state, starter],
            outputs=_flow_outputs(
                profile_state,
                action_state,
                assistant_message,
                idea_cards,
                style_cards,
                symbol_cards,
                contrast_cards,
                visual_group,
                debug_reason,
            ),
        )
        surprise.click(
            fn=_surprise,
            inputs=profile_state,
            outputs=_flow_outputs(
                profile_state,
                action_state,
                assistant_message,
                idea_cards,
                style_cards,
                symbol_cards,
                contrast_cards,
                visual_group,
                debug_reason,
            ),
        )
        continue_button.click(
            fn=_continue,
            inputs=[profile_state, action_state, idea_cards, style_cards, symbol_cards, contrast_cards],
            outputs=_flow_outputs(
                profile_state,
                action_state,
                assistant_message,
                idea_cards,
                style_cards,
                symbol_cards,
                contrast_cards,
                visual_group,
                debug_reason,
            ),
        )
        visual_submit.click(
            fn=_submit_visuals,
            inputs=[profile_state, minimal_rich, calm_intense, geometric_organic],
            outputs=_flow_outputs(
                profile_state,
                action_state,
                assistant_message,
                idea_cards,
                style_cards,
                symbol_cards,
                contrast_cards,
                visual_group,
                debug_reason,
            ),
        )
        create_now.click(
            fn=_generate,
            inputs=[profile_state, recipe_state],
            outputs=[
                image_output,
                gallery_label,
                symbol_map,
                image_prompt,
                regen,
                regen_button,
                trace_button,
                recipe_state,
                model_activity,
            ],
        )
        regen_button.click(
            fn=_regenerate,
            inputs=[profile_state, recipe_state, regen],
            outputs=[
                image_output,
                gallery_label,
                symbol_map,
                image_prompt,
                regen,
                regen_button,
                trace_button,
                recipe_state,
                model_activity,
            ],
        )
        trace_button.click(
            fn=_export_trace,
            outputs=trace_file,
        )
        reset_button.click(
            fn=_reset,
            outputs=[
                profile_state,
                action_state,
                assistant_message,
                idea_cards,
                style_cards,
                symbol_cards,
                contrast_cards,
                visual_group,
                debug_reason,
                image_output,
                gallery_label,
                symbol_map,
                image_prompt,
                regen,
                regen_button,
                trace_button,
                trace_file,
                recipe_state,
                model_activity,
            ],
        )

    return demo


def _flow_outputs(*components):
    return list(components)


def _format_runtime_status() -> str:
    status = runtime_status()
    mode = status["mode"]
    if mode == "llamacpp":
        model = f'{status["llamacpp_repo_id"]} / {status["llamacpp_filename"]}'
    elif mode == "local":
        model = f'{status["openai_compatible_model"]} at {status["openai_compatible_base_url"]}'
    else:
        model = "deterministic fallback"
    return f"Runtime: `{mode}` | Text model: `{model}`"


def _profile(data: dict) -> InterviewProfile:
    return InterviewProfile.model_validate(data)


def _render_next(profile: InterviewProfile):
    if MODEL_UI_DIRECTOR_ENABLED:
        next_ui = choose_next_ui_with_model(profile)
        director_source = "model_enabled"
    else:
        next_ui = choose_next_ui(profile)
        director_source = "deterministic_fast_path"
    log_event("next_ui", {"profile": profile.model_dump(), "next_ui": next_ui.model_dump()})
    log_event("ui_director_mode", {"source": director_source})
    return [
        profile.model_dump(),
        next_ui.next_action.value,
        f"{next_ui.assistant_message}\n\n**{next_ui.question}**",
        gr.update(choices=next_ui.options, value=[], visible=next_ui.next_action == UIAction.SHOW_IDEA_CARDS),
        gr.update(choices=next_ui.options, value=None, visible=next_ui.next_action == UIAction.SHOW_STYLE_CARDS),
        gr.update(choices=next_ui.options, value=None, visible=next_ui.next_action == UIAction.SHOW_SYMBOL_CARDS),
        gr.update(choices=next_ui.options, value=None, visible=next_ui.next_action == UIAction.ASK_CONTRAST),
        gr.update(visible=next_ui.next_action == UIAction.SHOW_VISUAL_SLIDERS),
        f"`{next_ui.reason}`",
    ]


def _submit_free_text(data: dict, text: str):
    profile = apply_answer(_profile(data), UIAction.ASK_FREE_TEXT, text)
    return _render_next(profile)


def _submit_starter(data: dict, selected: str | None):
    text = selected.split(": ", 1)[1] if selected and ": " in selected else (selected or "")
    profile = apply_answer(_profile(data), UIAction.SHOW_IDEA_CARDS, text)
    return _render_next(profile)


def _surprise(data: dict):
    profile = start_with_surprise(_profile(data))
    return _render_next(profile)


def _continue(data: dict, action: str, ideas: list[str], style: str | None, symbol: str | None, contrast: str | None):
    selected = ""
    if action == UIAction.SHOW_IDEA_CARDS and ideas:
        selected = " | ".join(ideas)
    elif action == UIAction.SHOW_STYLE_CARDS:
        selected = style or ""
    elif action == UIAction.SHOW_SYMBOL_CARDS:
        selected = symbol or ""
    elif action == UIAction.ASK_CONTRAST:
        selected = contrast or ""
    profile = apply_answer(_profile(data), action, selected)
    return _render_next(profile)


def _submit_visuals(data: dict, minimal_rich: float, calm_intense: float, geometric_organic: float):
    profile = apply_answer(
        _profile(data),
        UIAction.SHOW_VISUAL_SLIDERS,
        "visual preferences",
        {
            "minimal_rich": minimal_rich,
            "calm_intense": calm_intense,
            "geometric_organic": geometric_organic,
        },
    )
    return _render_next(profile)


@spaces.GPU(duration=180)
def _generate(data: dict, recipe_data: dict | None):
    profile = _profile(data)
    yield _pending_outputs(
        "Starting generation...\n"
        f"Runtime mode: {runtime_status()['mode']}\n"
        "Calling the art director. The first llama.cpp call may include model download/load time.",
        recipe_data,
    )
    recipe = build_art_recipe_with_model(profile)
    yield _pending_outputs(
        "Art recipe ready.\n"
        f"Title: {recipe.title}\n"
        "Rendering fallback painting now.",
        recipe.model_dump(),
    )
    image_result = ImageClient().generate(recipe)
    log_event(
        "image_generation",
        {
            "source": image_result.source,
            "elapsed_seconds": image_result.elapsed_seconds,
            "error": image_result.error,
            "image_path": image_result.path,
        },
    )
    log_event("generate", {"profile": profile.model_dump(), "recipe": recipe.model_dump(), "image_path": image_result.path})
    yield _gallery_outputs(
        image_result.path,
        recipe,
        f"Generation complete.\nImage source: {image_result.source}\nDownload the trace to confirm model source and fallback status.",
    )


@spaces.GPU(duration=180)
def _regenerate(data: dict, recipe_data: dict | None, instruction: str | None):
    profile = _profile(data)
    yield _pending_outputs(
        "Starting regeneration...\n"
        f"Instruction: {instruction or 'Surprise me'}\n"
        f"Runtime mode: {runtime_status()['mode']}",
        recipe_data,
    )
    recipe = build_art_recipe_with_model(profile, instruction or "Surprise me")
    yield _pending_outputs(
        "Updated recipe ready.\n"
        f"Title: {recipe.title}\n"
        "Rendering fallback painting now.",
        recipe.model_dump(),
    )
    image_result = ImageClient().generate(recipe)
    log_event(
        "image_generation",
        {
            "source": image_result.source,
            "elapsed_seconds": image_result.elapsed_seconds,
            "error": image_result.error,
            "image_path": image_result.path,
        },
    )
    log_event(
        "regenerate",
        {
            "profile": profile.model_dump(),
            "instruction": instruction,
            "recipe": recipe.model_dump(),
            "image_path": image_result.path,
        },
    )
    yield _gallery_outputs(
        image_result.path,
        recipe,
        f"Regeneration complete.\nImage source: {image_result.source}\nDownload the trace to confirm model source and fallback status.",
    )


def _pending_outputs(activity: str, recipe_data: dict | None):
    return [
        None,
        "### Working...",
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        recipe_data,
        gr.update(value=activity, visible=True),
    ]


def _gallery_outputs(path: str, recipe, activity: str):
    label = f"## {recipe.title}\n\n{recipe.friend_explanation}"
    symbol_rows = "\n".join(f"| {symbol.visual} | {symbol.meaning} |" for symbol in recipe.symbols)
    symbols = f"| Visual element | Meaning |\n|---|---|\n{symbol_rows}"
    return [
        path,
        label,
        symbols,
        gr.update(value=recipe.image_prompt, visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        recipe.model_dump(),
        gr.update(value=activity, visible=True),
    ]


def _export_trace():
    return gr.update(value=export_trace(), visible=True)


def _reset():
    profile = new_profile()
    return [
        profile.model_dump(),
        UIAction.ASK_FREE_TEXT.value,
        "Choose a starting idea, write your own, or ask the atelier to surprise you.",
        gr.update(choices=[], value=[], visible=False),
        gr.update(choices=[], value=None, visible=False),
        gr.update(choices=[], value=None, visible=False),
        gr.update(choices=[], value=None, visible=False),
        gr.update(visible=False),
        "",
        None,
        "",
        "",
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        None,
        gr.update(value="", visible=False),
    ]
