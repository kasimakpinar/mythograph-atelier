import gradio as gr
import spaces

from mythograph.config import APP_TITLE, IMAGE_HEIGHT, IMAGE_WIDTH, ROOT_DIR
from mythograph.models.image_client import ImageClient
from mythograph.models.llm_client import LlamaCppClient, preload_llamacpp_if_configured, runtime_status
from mythograph.schemas.profile import InterviewProfile
from mythograph.schemas.ui import ConversationStarter, ControlKind, ControlResponse, ConversationTurn
from mythograph.services.art_recipe import build_art_recipe_with_model
from mythograph.services.conversation import advance_conversation, model_error_turn, new_profile, start_session
from mythograph.services.starters import generate_conversation_starters
from mythograph.services.trace_logger import export_trace, log_event
from mythograph.ui.examples import REGENERATION_OPTIONS
from mythograph.ui.waiting_gallery import render_waiting_gallery


STARTER_LOADING_LABEL = "Model is loading..."
STARTER_COUNT = 3


def build_demo() -> gr.Blocks:
    css = (ROOT_DIR / "mythograph" / "ui" / "styles.css").read_text(encoding="utf-8")
    initial_turn = start_session()
    preload_result = preload_llamacpp_if_configured()
    if preload_result:
        log_event(
            "llamacpp_preload",
            {
                "source": preload_result.source,
                "error": preload_result.error,
                "runtime": runtime_status(),
            },
        )

    with gr.Blocks(title=APP_TITLE, css=css, elem_id="ma-shell", theme=gr.themes.Base()) as demo:
        profile_state = gr.State(new_profile().model_dump())
        chat_history_state = gr.State([])
        turn_state = gr.State(initial_turn.model_dump())
        recipe_state = gr.State(None)
        starter_state = gr.State([])
        starter_index_state = gr.State(0)
        generate_action_state = gr.State("generate")
        regenerate_action_state = gr.State("regenerate")

        with gr.Group(elem_id="ma-start-panel") as start_panel:
            gr.HTML(
                """
                <section class="ma-start">
                  <div class="ma-kicker">Mythograph Atelier</div>
                  <h1>Tell the painting what it has to mean.</h1>
                  <p>The atelier will ask only what it needs, then turn your answers into abstract art with a story you can say out loud.</p>
                </section>
                """,
                container=False,
            )
            start_prompt = gr.Textbox(
                label="",
                placeholder="I want something about ambition, patience, and becoming stronger slowly.",
                lines=1,
                max_lines=1,
                elem_id="ma-start-input",
                show_label=False,
            )
            with gr.Row(elem_classes=["ma-start-actions"]):
                start_submit = gr.Button("Begin", variant="primary", elem_classes=["ma-send"])
                refresh_starters = gr.Button("Refresh examples", elem_classes=["ma-ghost"])
                reset_from_start = gr.Button("Reset", elem_classes=["ma-ghost"])
            with gr.Row(elem_classes=["ma-starters"]):
                starter_button = gr.Button(
                    STARTER_LOADING_LABEL,
                    elem_classes=["ma-chip", "ma-rotating-starter"],
                    size="sm",
                    interactive=False,
                )
                starter_timer = gr.Timer(value=4, active=True)

        with gr.Group(visible=False, elem_id="ma-chat-panel") as chat_panel:
            with gr.Row(elem_classes=["ma-topbar"]):
                gr.Markdown("## Mythograph Atelier", elem_classes=["ma-brand"])
                gr.Markdown(_format_runtime_status(), elem_classes=["ma-runtime"])

            chatbot = gr.Chatbot(
                value=[],
                type="messages",
                label="",
                show_label=False,
                height=430,
                elem_id="ma-chatbot",
                avatar_images=(None, None),
                bubble_full_width=False,
            )
            progress = gr.Markdown(initial_turn.progress_label, elem_classes=["ma-progress"])
            waiting_gallery = gr.HTML("", visible=False, elem_classes=["ma-waiting-shell"], container=False)

            with gr.Group(elem_id="ma-control-tray"):
                with gr.Group(visible=False, elem_classes=["ma-control-group"]) as choice_group:
                    choice_cards = gr.Radio(label="", choices=[], show_label=False, elem_classes=["ma-card-control"])
                    choice_submit = gr.Button("Choose", variant="primary")

                with gr.Group(visible=False, elem_classes=["ma-control-group"]) as multi_group:
                    multi_cards = gr.CheckboxGroup(label="", choices=[], show_label=False, elem_classes=["ma-card-control"])
                    multi_submit = gr.Button("Choose", variant="primary")

                with gr.Group(visible=False, elem_classes=["ma-control-group"]) as slider_group:
                    gr.Markdown("### Studio dials")
                    minimal_rich = gr.Slider(0, 100, value=35, step=1, label="minimal to rich")
                    calm_intense = gr.Slider(0, 100, value=45, step=1, label="calm to intense")
                    geometric_organic = gr.Slider(0, 100, value=45, step=1, label="geometric to organic")
                    slider_submit = gr.Button("Save taste", variant="primary")

                with gr.Group(visible=False, elem_classes=["ma-control-group"]) as swatch_group:
                    swatch_picker = gr.Radio(label="", choices=[], show_label=False, elem_classes=["ma-swatch-control"])
                    swatch_submit = gr.Button("Set color weather", variant="primary")

                with gr.Group(visible=False, elem_classes=["ma-control-group"]) as refine_group:
                    refine_text = gr.Textbox(
                        label="",
                        placeholder="Add one sentence, or name a feeling, symbol, memory, or contradiction.",
                        lines=1,
                        max_lines=1,
                        show_label=False,
                    )
                    refine_submit = gr.Button("Add this", variant="primary")

                ready_button = gr.Button("Create artwork", variant="primary", visible=False, elem_id="ma-create-button")

            with gr.Row(elem_id="ma-input-dock"):
                chat_text = gr.Textbox(
                    label="",
                    placeholder="Answer in your own words, or add a thought, correction, quote, or memory...",
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    scale=5,
                )
                chat_submit = gr.Button("Send", variant="primary", scale=1, elem_classes=["ma-send"])
                reset_button = gr.Button("Reset", scale=1, elem_classes=["ma-ghost"])

            with gr.Accordion("Model activity", open=False):
                model_activity = gr.Textbox(
                    label="Activity",
                    lines=7,
                    visible=False,
                    interactive=False,
                )
                trace_button = gr.Button("Download trace", visible=False)
                trace_file = gr.File(label="Trace export", visible=False)

            with gr.Group(visible=False, elem_id="ma-gallery") as gallery_group:
                with gr.Row(elem_classes=["ma-gallery-grid"]):
                    with gr.Column(scale=5):
                        image_output = gr.Image(
                            label="",
                            type="filepath",
                            height=_gallery_image_height(),
                            show_label=False,
                        )
                    with gr.Column(scale=4, elem_classes=["ma-result-copy"]):
                        gallery_label = gr.Markdown("")
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

        flow_outputs = [
            profile_state,
            chat_history_state,
            turn_state,
            start_panel,
            chat_panel,
            chatbot,
            chat_text,
            progress,
            waiting_gallery,
            choice_group,
            choice_cards,
            multi_group,
            multi_cards,
            slider_group,
            minimal_rich,
            calm_intense,
            geometric_organic,
            swatch_group,
            swatch_picker,
            refine_group,
            refine_text,
            ready_button,
            gallery_group,
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
        ]

        start_submit.click(
            fn=_submit_text,
            inputs=[profile_state, chat_history_state, start_prompt],
            outputs=flow_outputs,
        )
        start_prompt.submit(
            fn=_submit_text,
            inputs=[profile_state, chat_history_state, start_prompt],
            outputs=flow_outputs,
        )
        chat_submit.click(
            fn=_submit_text,
            inputs=[profile_state, chat_history_state, chat_text],
            outputs=flow_outputs,
        )
        chat_text.submit(
            fn=_submit_text,
            inputs=[profile_state, chat_history_state, chat_text],
            outputs=flow_outputs,
        )
        refine_submit.click(
            fn=_submit_text,
            inputs=[profile_state, chat_history_state, refine_text],
            outputs=flow_outputs,
        )
        refine_text.submit(
            fn=_submit_text,
            inputs=[profile_state, chat_history_state, refine_text],
            outputs=flow_outputs,
        )

        starter_button.click(
            fn=_submit_starter,
            inputs=[profile_state, chat_history_state, starter_state, starter_index_state],
            outputs=flow_outputs,
        )

        refresh_starters.click(
            fn=_refresh_starters,
            inputs=[starter_state],
            outputs=[starter_state, starter_index_state, starter_button],
        )

        demo.load(
            fn=_refresh_starters,
            inputs=[starter_state],
            outputs=[starter_state, starter_index_state, starter_button],
            show_progress="hidden",
        )
        starter_timer.tick(
            fn=_rotate_starter,
            inputs=[starter_state, starter_index_state],
            outputs=[starter_index_state, starter_button],
            show_progress="hidden",
        )

        choice_submit.click(
            fn=_submit_choice,
            inputs=[profile_state, chat_history_state, turn_state, choice_cards],
            outputs=flow_outputs,
        )
        multi_submit.click(
            fn=_submit_multi,
            inputs=[profile_state, chat_history_state, turn_state, multi_cards],
            outputs=flow_outputs,
        )
        slider_submit.click(
            fn=_submit_sliders,
            inputs=[profile_state, chat_history_state, turn_state, minimal_rich, calm_intense, geometric_organic],
            outputs=flow_outputs,
        )
        swatch_submit.click(
            fn=_submit_swatch,
            inputs=[profile_state, chat_history_state, turn_state, swatch_picker],
            outputs=flow_outputs,
        )
        ready_event = ready_button.click(
            fn=_prepare_generation,
            inputs=[profile_state, chat_history_state, recipe_state],
            outputs=flow_outputs,
        )
        ready_event.then(
            fn=_render_prepared_image,
            inputs=[profile_state, chat_history_state, recipe_state, generate_action_state],
            outputs=flow_outputs,
        )
        regen_event = regen_button.click(
            fn=_prepare_regeneration,
            inputs=[profile_state, chat_history_state, recipe_state, regen],
            outputs=flow_outputs,
        )
        regen_event.then(
            fn=_render_prepared_image,
            inputs=[profile_state, chat_history_state, recipe_state, regenerate_action_state],
            outputs=flow_outputs,
        )
        trace_button.click(fn=_export_trace, outputs=trace_file)

        reset_button.click(fn=_reset, outputs=flow_outputs).then(
            fn=_refresh_starters,
            inputs=[starter_state],
            outputs=[starter_state, starter_index_state, starter_button],
            show_progress="hidden",
        )
        reset_from_start.click(fn=_reset, outputs=flow_outputs).then(
            fn=_refresh_starters,
            inputs=[starter_state],
            outputs=[starter_state, starter_index_state, starter_button],
            show_progress="hidden",
        )

    return demo


def _format_runtime_status() -> str:
    status = runtime_status()
    mode = status["mode"]
    if mode == "llamacpp":
        model = (
            f'{status["llamacpp_repo_id"]} / {status["llamacpp_filename"]} '
            f'(CPU threads: {status["llamacpp_n_threads"]}, preload: {status["llamacpp_preload"]}, '
            f'GPU layers: {status["llamacpp_n_gpu_layers"]}, thinking: {status["llamacpp_recipe_thinking"]}, '
            f'unload: {status["llamacpp_unload_after_call"]})'
        )
    elif mode == "local":
        model = f'{status["openai_compatible_model"]} at {status["openai_compatible_base_url"]}'
    else:
        model = "unavailable"
    return f"Runtime: `{mode}` | Text model: `{model}`"


def _gallery_image_height() -> int:
    if IMAGE_WIDTH <= 0 or IMAGE_HEIGHT <= 0:
        return 620
    return max(360, min(760, int(620 * (IMAGE_HEIGHT / IMAGE_WIDTH))))


def _profile(data: dict) -> InterviewProfile:
    return InterviewProfile.model_validate(data)


def _turn(data: dict) -> ConversationTurn:
    return ConversationTurn.model_validate(data)


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _submit_text(profile_data: dict, history: list[dict], text: str):
    clean = (text or "").strip()
    if not clean:
        yield _reset()
        return
    director_history = (history or []) + [_message("user", clean)]
    yield _thinking_render(_profile(profile_data), director_history, _thinking_label(history))
    profile_data, turn_data = _advance_conversation_on_gpu(profile_data, clean, None, director_history)
    profile = _profile(profile_data)
    turn = _turn(turn_data)
    chat = director_history + [_message("assistant", turn.assistant_message)]
    yield _render(profile, chat, turn, activity=_activity_for_turn(turn, "GPU text model done."))


def _submit_starter(profile_data: dict, history: list[dict], starters_data: list[dict], index: int):
    starters = _starter_models(starters_data)
    if not starters:
        yield _reset()
        return
    starter = starters[index % len(starters)]
    text = starter.text
    director_history = (history or []) + [_message("user", text)]
    yield _thinking_render(_profile(profile_data), director_history, _thinking_label(history))
    profile_data, turn_data = _advance_conversation_on_gpu(profile_data, text, None, director_history)
    profile = _profile(profile_data)
    turn = _turn(turn_data)
    chat = director_history + [_message("assistant", turn.assistant_message)]
    yield _render(profile, chat, turn, activity=_activity_for_turn(turn, "GPU text model done."))


@spaces.GPU(duration=60)
def _refresh_starters(starters_data: list[dict]):
    try:
        starters = generate_conversation_starters(count=STARTER_COUNT)
    except Exception as exc:
        log_event("llm_conversation_starters", {"source": "error", "error": str(exc), "retry_count": 0})
        return [], 0, gr.update(value="Starter generation failed. Type your own first sentence.", interactive=False)
    data = [starter.model_dump() for starter in starters]
    return data, 0, _starter_button_update(starters, 0)


def _rotate_starter(starters_data: list[dict] | None, index: int | None):
    starters = _starter_models(starters_data)
    if not starters:
        return 0, gr.update(value=STARTER_LOADING_LABEL, interactive=False)
    next_index = ((index or 0) + 1) % len(starters)
    return next_index, _starter_button_update(starters, next_index)


def _starter_button_update(starters: list[ConversationStarter], index: int):
    if not starters:
        return gr.update(value=STARTER_LOADING_LABEL, interactive=False)
    starter = starters[index % len(starters)]
    return gr.update(value=starter.text, interactive=True)


def _starter_models(starters_data: list[dict] | None) -> list[ConversationStarter]:
    starters: list[ConversationStarter] = []
    for item in starters_data or []:
        try:
            starters.append(ConversationStarter.model_validate(item))
        except Exception:
            continue
    return starters


def _submit_choice(profile_data: dict, history: list[dict], turn_data: dict, value: str | None):
    control = _current_control(turn_data)
    response = ControlResponse(
        kind=ControlKind.CHOICE_CARDS,
        values=[value] if value else [],
        label=control.label if control else "",
        prompt=control.prompt if control else "",
    )
    yield from _submit_control(profile_data, history, response, value or "I am not sure yet.")


def _submit_multi(profile_data: dict, history: list[dict], turn_data: dict, values: list[str] | None):
    control = _current_control(turn_data)
    chosen = values or []
    response = ControlResponse(
        kind=ControlKind.MULTI_CHOICE_CARDS,
        values=chosen,
        label=control.label if control else "",
        prompt=control.prompt if control else "",
    )
    yield from _submit_control(profile_data, history, response, " | ".join(chosen) if chosen else "I am not sure yet.")


def _submit_sliders(
    profile_data: dict,
    history: list[dict],
    turn_data: dict,
    minimal_rich: float,
    calm_intense: float,
    geometric_organic: float,
):
    control = _current_control(turn_data)
    values = [minimal_rich, calm_intense, geometric_organic]
    sliders = _slider_response_values(control, values)
    response = ControlResponse(
        kind=ControlKind.SLIDER_GROUP,
        sliders=sliders,
        label=control.label if control else "",
        prompt=control.prompt if control else "",
    )
    summary = _slider_summary(control, sliders)
    yield from _submit_control(profile_data, history, response, summary)


def _submit_swatch(profile_data: dict, history: list[dict], turn_data: dict, value: str | None):
    control = _current_control(turn_data)
    response = ControlResponse(
        kind=ControlKind.SWATCH_PICKER,
        values=[value] if value else [],
        label=control.label if control else "",
        prompt=control.prompt if control else "",
    )
    yield from _submit_control(profile_data, history, response, value or "Surprise me.")


def _current_control(turn_data: dict):
    try:
        turn = _turn(turn_data)
    except Exception:
        return None
    return turn.controls[0] if turn.controls else None


def _slider_response_values(control, values: list[float]) -> dict[str, float]:
    if not control or not control.sliders:
        return {
            "minimal_rich": values[0],
            "calm_intense": values[1],
            "geometric_organic": values[2],
        }
    sliders: dict[str, float] = {}
    for index, slider in enumerate(control.sliders[:3]):
        sliders[slider.key] = values[index]
    return sliders


def _slider_summary(control, sliders: dict[str, float]) -> str:
    if not control or not control.sliders:
        return (
            f"Density {sliders.get('minimal_rich', 0):.0f}, "
            f"energy {sliders.get('calm_intense', 0):.0f}, "
            f"shape {sliders.get('geometric_organic', 0):.0f}."
        )
    parts = []
    for slider in control.sliders[:3]:
        value = sliders.get(slider.key, slider.value)
        parts.append(f"{slider.label} {value:.0f}")
    return ", ".join(parts) + "."


def _submit_control(profile_data: dict, history: list[dict], response: ControlResponse, user_summary: str):
    director_history = (history or []) + [_message("user", user_summary)]
    yield _thinking_render(_profile(profile_data), director_history, _thinking_label(history))
    profile_data, turn_data = _advance_conversation_on_gpu(profile_data, "", response.model_dump(), director_history)
    profile = _profile(profile_data)
    turn = _turn(turn_data)
    chat = director_history + [_message("assistant", turn.assistant_message)]
    yield _render(profile, chat, turn, activity=_activity_for_turn(turn, "GPU text model done."))


def _thinking_render(profile: InterviewProfile, history: list[dict], progress_label: str):
    if progress_label.startswith("Loading"):
        activity = (
            "Loading GPU text model...\n"
            "The first reply can take longer while llama.cpp loads the GGUF. Later replies should be faster."
        )
        waiting_html = render_waiting_gallery("The model is loading now. Meanwhile, you can take a look at the gallery.")
    else:
        activity = "GPU text model thinking...\nNemotron is choosing the next question and control."
        waiting_html = ""
    return _render(
        profile,
        history,
        ConversationTurn(
            assistant_message=progress_label,
            progress_label=progress_label,
            reason="model-assisted turn is running",
            is_ready=False,
        ),
        activity=activity,
        waiting_html=waiting_html,
    )


def _thinking_label(history: list[dict] | None) -> str:
    return "Loading GPU text model..." if not history else "GPU text model thinking..."


def _activity_for_turn(turn: ConversationTurn, default: str) -> str:
    if turn.reason.startswith("Text model error"):
        return f"{turn.assistant_message}\n{turn.reason}"
    return default


@spaces.GPU(duration=60)
def _advance_conversation_on_gpu(
    profile_data: dict,
    user_message: str,
    control_response_data: dict | None,
    conversation_history: list[dict] | None = None,
):
    profile = _profile(profile_data)
    try:
        response = ControlResponse.model_validate(control_response_data) if control_response_data else None
        profile, turn = advance_conversation(
            profile,
            user_message=user_message,
            control_response=response,
            conversation_history=conversation_history,
        )
        return profile.model_dump(), turn.model_dump()
    except Exception as exc:
        log_event(
            "llm_conversation_turn",
            {
                "source": "error",
                "error": f"ZeroGPU llama.cpp turn failed: {exc}",
                "retry_count": 0,
            },
        )
        return profile.model_dump(), model_error_turn("The GPU text worker failed. Try again once.", str(exc)).model_dump()


def _render(
    profile: InterviewProfile,
    history: list[dict],
    turn: ConversationTurn,
    recipe_data: dict | None = None,
    activity: str = "",
    image_path: str | None = None,
    recipe=None,
    show_gallery: bool = False,
    waiting_html: str = "",
):
    control = turn.controls[0] if turn.controls else None
    kind = control.kind if control else None
    options = control.options if control else []

    label = ""
    symbols = ""
    prompt = gr.update(visible=False)
    regen = gr.update(visible=False)
    regen_button = gr.update(visible=False)
    trace_button = gr.update(visible=bool(history) or bool(activity))
    model_activity = gr.update(value=activity, visible=bool(activity))
    waiting_gallery = gr.update(value=waiting_html, visible=bool(waiting_html))
    chat_placeholder = "Answer in your own words, or add a thought, correction, quote, or memory..."
    if kind == ControlKind.TEXT_REFINEMENT and control and control.prompt:
        chat_placeholder = control.prompt
    if recipe:
        central_phrase = getattr(recipe, "central_phrase", "").strip()
        phrase_block = f"\n\n> {central_phrase}\n" if central_phrase else ""
        label = f"## {recipe.title}{phrase_block}\n\n{recipe.friend_explanation}"
        symbols = _meaning_links(recipe)
        prompt = gr.update(value=recipe.image_prompt, visible=True)
        regen = gr.update(visible=True)
        regen_button = gr.update(visible=True)
        trace_button = gr.update(visible=True)

    return [
        profile.model_dump(),
        history,
        turn.model_dump(),
        gr.update(visible=False),
        gr.update(visible=True),
        history,
        gr.update(value="", placeholder=chat_placeholder),
        f"**{turn.progress_label}**",
        waiting_gallery,
        gr.update(visible=kind == ControlKind.CHOICE_CARDS),
        gr.update(choices=options, value=None),
        gr.update(visible=kind == ControlKind.MULTI_CHOICE_CARDS),
        gr.update(choices=options, value=[]),
        gr.update(visible=kind == ControlKind.SLIDER_GROUP),
        gr.update(value=_slider_value_at(turn, 0, 35), label=_slider_label_at(turn, 0, "minimal to rich")),
        gr.update(value=_slider_value_at(turn, 1, 45), label=_slider_label_at(turn, 1, "calm to intense")),
        gr.update(value=_slider_value_at(turn, 2, 45), label=_slider_label_at(turn, 2, "geometric to organic")),
        gr.update(visible=kind == ControlKind.SWATCH_PICKER),
        gr.update(choices=options, value=None),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=turn.is_ready),
        gr.update(visible=show_gallery),
        image_path,
        label,
        symbols,
        prompt,
        regen,
        regen_button,
        trace_button,
        gr.update(value=None, visible=False),
        recipe_data,
        model_activity,
    ]


def _slider_value(turn: ConversationTurn, key: str, default: int) -> int:
    if not turn.controls:
        return default
    for slider in turn.controls[0].sliders:
        if slider.key == key:
            return slider.value
    return default


def _slider_value_at(turn: ConversationTurn, index: int, default: int) -> int:
    if not turn.controls or len(turn.controls[0].sliders) <= index:
        return default
    return turn.controls[0].sliders[index].value


def _slider_label_at(turn: ConversationTurn, index: int, default: str) -> str:
    if not turn.controls or len(turn.controls[0].sliders) <= index:
        return default
    slider = turn.controls[0].sliders[index]
    label = slider.label.strip() or default
    left = slider.left_label.strip()
    right = slider.right_label.strip()
    if left and right:
        return f"{label}: {left} to {right}"
    return label


def _meaning_links(recipe) -> str:
    if not getattr(recipe, "symbols", None):
        return ""
    lines = ["### Meaning links"]
    for symbol in recipe.symbols:
        lines.append(f"- **{symbol.meaning}** becomes **{symbol.visual}**.")
    return "\n".join(lines)


@spaces.GPU(duration=120)
def _prepare_generation(profile_data: dict, history: list[dict], recipe_data: dict | None):
    profile = _profile(profile_data)
    working_history = (history or []) + [_message("assistant", "I am preparing the art recipe with the GPU text model.")]
    turn = ConversationTurn(
        assistant_message="I am preparing the art recipe with the GPU text model.",
        progress_label="Painting: art director is working",
        reason="generation started",
        is_ready=False,
    )
    yield _render(
        profile,
        working_history,
        turn,
        recipe_data,
        "GPU text recipe is being prepared.",
        waiting_html=render_waiting_gallery("The art recipe is being written now. Meanwhile, you can take a look at the gallery."),
    )
    try:
        recipe = build_art_recipe_with_model(profile)
    except Exception as exc:
        error_turn = model_error_turn("The recipe model failed after retries. Please try again.", str(exc))
        yield _render(
            profile,
            working_history + [_message("assistant", error_turn.assistant_message)],
            error_turn,
            recipe_data,
            f"{error_turn.assistant_message}\n{error_turn.reason}",
        )
        return
    yield _render(
        profile,
        working_history,
        turn,
        recipe.model_dump(),
        "GPU text recipe ready.\n"
        f"Runtime mode: {runtime_status()['mode']}\n"
        f"Title: {recipe.title}\n"
        "Unloading llama.cpp before requesting FLUX.",
    )


@spaces.GPU(duration=120)
def _prepare_regeneration(profile_data: dict, history: list[dict], recipe_data: dict | None, instruction: str | None):
    profile = _profile(profile_data)
    working_history = (history or []) + [_message("assistant", "I am revising the recipe with the GPU text model.")]
    turn = ConversationTurn(
        assistant_message="I am revising the recipe with the GPU text model.",
        progress_label="Painting: art director is revising",
        reason="regeneration started",
        is_ready=False,
    )
    yield _render(
        profile,
        working_history,
        turn,
        recipe_data,
        "GPU text recipe revision is being prepared.",
        waiting_html=render_waiting_gallery("The recipe is being revised now. Meanwhile, you can take a look at the gallery."),
    )
    try:
        recipe = build_art_recipe_with_model(profile, instruction or "Surprise me")
    except Exception as exc:
        error_turn = model_error_turn("The recipe model failed after retries. Please try again.", str(exc))
        yield _render(
            profile,
            working_history + [_message("assistant", error_turn.assistant_message)],
            error_turn,
            recipe_data,
            f"{error_turn.assistant_message}\n{error_turn.reason}",
        )
        return
    yield _render(
        profile,
        working_history,
        turn,
        recipe.model_dump(),
        "GPU text recipe revision ready.\n"
        f"Instruction: {instruction or 'Surprise me'}\n"
        f"Title: {recipe.title}\n"
        "Unloading llama.cpp before requesting FLUX.",
    )


@spaces.GPU(duration=300)
def _render_prepared_image(
    profile_data: dict,
    history: list[dict],
    recipe_data: dict | None,
    action: str = "generate",
):
    profile = _profile(profile_data)
    if not recipe_data:
        yield _render(
            profile,
            history or [],
            start_session(),
            recipe_data,
            "No recipe was prepared; please continue the conversation and try again.",
        )
        return
    from mythograph.schemas.art_recipe import ArtRecipe

    LlamaCppClient.unload()
    recipe = ArtRecipe.model_validate(recipe_data)
    yield _render(
        profile,
        history or [],
        ConversationTurn(
            assistant_message="I am rendering the image now.",
            progress_label="Painting: FLUX is rendering",
            reason="image generation in progress",
            is_ready=False,
        ),
        recipe.model_dump(),
        "FLUX image generation is running.",
        recipe=recipe,
        waiting_html=render_waiting_gallery("The image is being generated now. Meanwhile, you can take a look at the gallery."),
    )
    try:
        image_result = ImageClient().generate(recipe)
    except Exception as exc:
        log_event(
            "image_generation",
            {
                "source": "error",
                "error": str(exc),
                "image_path": None,
            },
        )
        error_turn = model_error_turn("Image generation failed. Please try again.", str(exc))
        yield _render(
            profile,
            (history or []) + [_message("assistant", error_turn.assistant_message)],
            error_turn,
            recipe.model_dump(),
            f"{error_turn.assistant_message}\n{error_turn.reason}",
            None,
            recipe,
            False,
        )
        return
    log_event(
        "image_generation",
        {
            "source": image_result.source,
            "elapsed_seconds": image_result.elapsed_seconds,
            "error": image_result.error,
            "image_path": image_result.path,
        },
    )
    event_name = "regenerate" if action == "regenerate" else "generate"
    log_event(event_name, {"profile": profile.model_dump(), "recipe": recipe.model_dump(), "image_path": image_result.path})
    final_history = (history or []) + [_message("assistant", f"Done. The piece is called **{recipe.title}**.")]
    yield _render(
        profile,
        final_history,
        ConversationTurn(
            assistant_message=f"Done. The piece is called {recipe.title}.",
            progress_label="Complete: image rendered",
            reason="image generation complete",
            is_ready=False,
        ),
        recipe.model_dump(),
        f"Generation complete.\nImage source: {image_result.source}\nDownload the trace to confirm model source and errors.",
        image_result.path,
        recipe,
        True,
    )


def _export_trace():
    return gr.update(value=export_trace(), visible=True)


def _reset():
    profile = new_profile()
    turn = start_session()
    return [
        profile.model_dump(),
        [],
        turn.model_dump(),
        gr.update(visible=True),
        gr.update(visible=False),
        [],
        gr.update(value=""),
        f"**{turn.progress_label}**",
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        gr.update(visible=False),
        gr.update(choices=[], value=[]),
        gr.update(visible=False),
        gr.update(value=35),
        gr.update(value=45),
        gr.update(value=45),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False),
        gr.update(visible=False),
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
