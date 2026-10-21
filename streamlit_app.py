import json

import streamlit as st

from agent import create_research_agent
from model import resultModel


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
SUGGESTIONS = (
    "What are the latest AI news headlines?",
    "Explain compound interest with an example",
    "What is 18% of 245?",
)
TOOL_LABELS = {
    "calculator": "Calculator",
    "duckduckgo_search": "Web search",
    "read_webpage": "Webpage reader",
}

st.set_page_config(
    page_title="Research assistant",
    page_icon=":material/travel_explore:",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(66, 101, 189, 0.13), transparent 32rem),
            var(--background-color);
    }
    main .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }
    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 1rem;
        padding: 1rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("connected_config", None)

with st.sidebar:
    st.title("Configure your model")
    st.caption("Your API settings stay in this browser session.")

    with st.form("api_settings"):
        api_key = st.text_input(
            "API key",
            type="password",
            help="Used only for requests from your current session.",
        )
        base_url = st.text_input(
            "API base URL",
            value=DEFAULT_BASE_URL,
            help="Use an OpenAI-compatible chat completions endpoint.",
        )
        model_name = st.text_input("Model", value=DEFAULT_MODEL)
        test_connection = st.form_submit_button(
            "Connect and test",
            type="primary",
            width="stretch",
        )

    config = (api_key.strip(), base_url.strip().rstrip("/"), model_name.strip())

    if test_connection:
        if not all(config):
            st.error("Enter an API key, base URL, and model name.")
            st.session_state.connected_config = None
        else:
            try:
                with st.spinner("Testing the model connection..."):
                    resultModel(
                        api_key=config[0],
                        base_url=config[1],
                        model_name=config[2],
                    ).invoke("Reply with only: OK")
            except Exception as exc:
                st.session_state.connected_config = None
                message = str(exc).replace(config[0], "[redacted]")
                st.error(f"Could not connect: {message}")
            else:
                st.session_state.connected_config = config
                st.success("Model connection is ready.")

    is_connected = st.session_state.connected_config == config and all(config)
    if is_connected:
        st.badge("Connected", icon=":material/check_circle:", color="green")
    else:
        st.caption("Connect and test your settings to start chatting.")

    if st.button("Clear conversation", width="stretch"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("### :material/travel_explore: Research assistant")
st.title("Ask. Explore. Understand.")
st.markdown(
    "A research assistant with web search, webpage reading, and calculator tools. "
    "Connect your own compatible model API in the sidebar to get started."
)

if not st.session_state.chat_history:
    st.markdown("##### Try a question")
    suggestion_columns = st.columns(len(SUGGESTIONS))
    selected_prompt = None
    for column, suggestion in zip(suggestion_columns, SUGGESTIONS):
        if column.button(suggestion, width="stretch", disabled=not is_connected):
            selected_prompt = suggestion
else:
    selected_prompt = None

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask a question...",
    disabled=not is_connected,
)
prompt = prompt or selected_prompt


def stream_text(chat_model, messages):
    for chunk in chat_model.stream(messages):
        content = chunk.content
        if isinstance(content, str):
            yield content
        elif content:
            yield str(content)


def format_activity_value(value, limit=1000):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    if len(value) > limit:
        return f"{value[:limit]}…"
    return value


if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        activity = None
        try:
            activity = st.status(
                "Agent is analyzing your request...",
                state="running",
                expanded=True,
            )
            activity.write("**Agent:** Research assistant")
            activity.write(f"**Model:** `{config[2]}`")
            activity.write("Reviewing the conversation and selecting the next step.")

            agent = create_research_agent(
                resultModel(
                    api_key=config[0],
                    base_url=config[1],
                    model_name=config[2],
                )
            )
            final_message = None
            tool_call_count = 0

            for update in agent.stream(
                {"messages": st.session_state.chat_history},
                stream_mode="updates",
            ):
                for node_update in update.values():
                    for message in node_update.get("messages", []):
                        tool_calls = getattr(message, "tool_calls", [])
                        if tool_calls:
                            for tool_call in tool_calls:
                                tool_call_count += 1
                                tool_name = tool_call.get("name", "unknown tool")
                                tool_label = TOOL_LABELS.get(
                                    tool_name,
                                    tool_name.replace("_", " ").title(),
                                )
                                activity.write(f"**Running tool:** {tool_label}")
                                arguments = tool_call.get("args")
                                if arguments:
                                    activity.write("Tool input:")
                                    activity.code(
                                        format_activity_value(arguments),
                                        language="json",
                                    )
                        elif getattr(message, "type", None) == "ai":
                            final_message = message

                        if getattr(message, "type", None) == "tool":
                            tool_name = getattr(message, "name", None) or "Tool"
                            tool_label = TOOL_LABELS.get(
                                tool_name,
                                tool_name.replace("_", " ").title(),
                            )
                            result = getattr(message, "content", None)
                            activity.write(
                                f"**{tool_label}:** "
                                f"{'Failed' if getattr(message, 'status', 'success') == 'error' else 'Finished'}"
                            )
                            if result:
                                activity.code(format_activity_value(result))

            if final_message is None:
                raise RuntimeError("The agent did not return a final answer.")
            if tool_call_count == 0:
                activity.write("No external tools were needed for this request.")

            raw_text = final_message.content
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)

            activity.update(
                label="Preparing the final answer...",
                state="running",
                expanded=True,
            )
            formatter = resultModel(
                api_key=config[0],
                base_url=config[1],
                model_name=config[2],
            )
            response = st.write_stream(
                stream_text(
                    formatter,
                    [
                        {
                            "role": "system",
                            "content": "Convert raw tool output into a clean, structured report.",
                        },
                        {
                            "role": "user",
                            "content": (
                                f"User question:\n{prompt}\n\n"
                                f"Raw tool output:\n{raw_text}\n\n"
                                "Write a clear answer with headings, bullet points, "
                                "and a short summary."
                            ),
                        },
                    ],
                )
            )
            if not response:
                raise RuntimeError("The model returned an empty answer.")
            activity.update(
                label="Request complete",
                state="complete",
                expanded=True,
            )
        except Exception as exc:
            if activity is not None:
                activity.update(
                    label="Request failed",
                    state="error",
                    expanded=True,
                )
            message = str(exc).replace(config[0], "[redacted]")
            response = f"Request failed: {message}"
            st.error(response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
