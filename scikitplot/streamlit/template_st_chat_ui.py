"""
Streamlit Conversational UI.

- https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
"""

# template_st_chat_ui.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=no-name-in-module
# pylint: disable=broad-exception-caught

import os
from typing import Optional

from scikitplot import logger
from scikitplot._compat.optional_deps import HAS_STREAMLIT, safe_import
from scikitplot.llm_provider import (
    LLM_MODEL_PROVIDER2ID,  # noqa: F401
    chat_provider,
    get_env_st_secrets,
    load_mlflow_gateway_config,
    load_st_secrets,
    save_st_secrets,
)

if HAS_STREAMLIT:
    st = safe_import("streamlit")

    @st.cache_resource
    def cached_config(path: str) -> "dict[str, any]":
        """cached_config."""
        return load_mlflow_gateway_config(path)

    def api_key_config_ui(
        config_path: "Optional[str]" = None,
    ) -> "tuple[str, str, Optional[str]]":
        """
        Render the Streamlit UI for API key configuration.

        Parameters
        ----------
        config_path : Optional[str]
            Path to a YAML config file in MLflow Gateway format (if available).

        Returns
        -------
        Tuple[str, str, Optional[str]]
            Selected model provider, model ID, and API key (if entered).
        """
        # st.subheader("🔐 API Key Configuration")
        st.subheader("\U0001f511 API Key Configuration")

        if "llm_model_provider2id" not in st.session_state:
            # pylint: disable=global-statement
            # pylint: disable=global-variable-not-assigned
            global LLM_MODEL_PROVIDER2ID  # noqa: PLW0602, PLW0603
            st.session_state.llm_model_provider2id = LLM_MODEL_PROVIDER2ID

        config_path = (
            config_path
            or st.text_input(
                "Get from MLflow gateway config file:\n( ./galleries/gateway/openai/config.yaml )",
                value="",
                placeholder="./galleries/gateway/openai/config.yaml",
                disabled=False,
                # on_change=call_func,
            ).strip()
        )
        # Load from config file if provided
        if config_path:
            try:
                # conf = load_mlflow_gateway_config(config_path)
                conf = cached_config(config_path)
                st.session_state.llm_model_provider2id = conf
            except Exception as e:
                # Fallback defaults
                # st.warning("No model configurations available.")
                st.error(f"Failed to load config: {e}")
                logger.error(f"Failed to load config: {e}")
        else:
            st.session_state.llm_model_provider2id = LLM_MODEL_PROVIDER2ID
        # Select provider
        model_provider = st.selectbox(
            label="Select Model Provider:",
            options=list(st.session_state.llm_model_provider2id.keys()),
            index=0,  # This default won't reapply if user changes selection
            help="Choose the model provider.",
        )
        # Select or input model ID
        model_options = [
            cfg["model_id"]
            for cfg in st.session_state.llm_model_provider2id[model_provider]
        ] + ["Custom..."]
        model_id = st.selectbox(
            label="Select or enter a model ID:",
            options=model_options,
            index=0,
            help="Choose a predefined model or enter a custom one.",
        )
        model_id = (
            st.text_input(
                label=f"Custom Model ID for {model_provider}",
                value="Model ID" if model_id != "Custom..." else "",
                key="custom_model_id",
                disabled=model_id != "Custom...",
            ).strip()
            if model_id == "Custom..."
            else model_id
        )
        key_default = next(
            (
                cfg["api_key"]
                for cfg in st.session_state.llm_model_provider2id[model_provider]
                if cfg["model_id"] == model_id
            ),
            "",
        )
        # API Key input
        key_input = st.text_input(
            f"Enter your {model_provider} API key:",
            value=key_default,
            type="password",
            placeholder=(
                "sk-..."
                if model_provider == "openai"
                else "hf_..." if model_provider == "huggingface" else "..."
            ),
            help="API key used to authenticate requests.",
        ).strip()
        # def valid_key_format(provider: str, key: str) -> bool:
        #     """valid_key_format."""
        #     return (
        #         (provider == "openai" and key.startswith("sk-")) or
        #         (provider == "huggingface" and key.startswith("hf_")) or
        #         (provider == "groq" and key.startswith("gsk_")) or
        #         (provider == "gemini" and key.startswith("gemini_")) or
        #         (provider == "anthropic" and key.startswith("sk-ant-")) or
        #         (provider == "cohere" and key.startswith("coh_"))
        #     )
        # product detection
        product = os.getenv("PRODUCT") or get_env_st_secrets("PRODUCT", "product")
        if st.button("Save API Key"):
            # if not key_input or len(key_input) < 10:
            #     st.warning("Please enter a valid API key.")
            # elif not valid_key_format(model_provider, key_input):
            #     st.warning(f"API key for {model_provider} must start with expected prefix.")
            # else:
            st.session_state[f"{model_provider}_api_key"] = key_input
            if product == "product":
                secrets = load_st_secrets()
                secrets[model_provider.upper() + "_API_KEY"] = key_input
                save_st_secrets(secrets)
                st.success(
                    f"{model_provider.capitalize()} API key saved and persisted!"
                )
            else:
                st.info("Dev mode: API key saved to session only.")
                st.success(f"{model_provider.capitalize()} API key saved in session!")

        # Show current key info (not the key itself)
        current_key = st.session_state.get(f"{model_provider}_api_key")
        if current_key:
            st.info(f"Loaded key for {model_provider} (hidden).")

        return model_provider, model_id, key_input if key_input else None

    def run_chat_ui():  # noqa: PLR0912
        """
        Render a simple chat interface using Streamlit with message history.

        Features:
        - Displays chat messages from session state.
        - Uses a bordered container to enclose chat history.
        - Input field stays at the bottom.
        - Echoes user input as a simulated assistant response.

        Session State Keys:
        - "messages": list of message dictionaries (role: 'user' or 'assistant', content: str)
        """
        # Placeholder
        with st.empty().container(border=True):
            # https://discuss.streamlit.io/t/how-to-right-justify-st-chat-message/46794/4
            st.markdown(
                """
            <style>
                .st-emotion-cache-janbn0 {
                    flex-direction: row-reverse;
                    text-align: right;
                }
            </style>
            """,
                unsafe_allow_html=True,
            )
            # Initialize chat history, append input + response to session state
            if "show_history" not in st.session_state:
                st.session_state.show_history = False
            if "messages" not in st.session_state:
                st.session_state.messages = []
            # Initialize the flag in session state
            if "running" not in st.session_state:
                st.session_state["running"] = False

            # Sidebar for controlling expanders and categories
            with st.sidebar:
                model_provider, model_id, key_input = api_key_config_ui()

            # with st.expander("💬 Assistant Chat"):
            # st.title("💬 ChatBot")
            st.subheader("💬 Assistant Chat")

            st.write(f"Selected model_type: {model_provider}")
            st.write(f"Selected model_id: {model_id}")

            # Placeholder
            chat_history_placeholder = st.empty().container()
            # Placeholder
            chat_messages_placeholder = st.empty().container(border=True)
            # Placeholder
            chat_btn_placeholder = st.empty().container()
            # Placeholder
            chat_btn_togg_placeholder = st.empty().container()

            # Fill Placeholders logically
            with chat_btn_togg_placeholder:
                # To place two buttons side-by-side (in the same horizontal row) in Streamlit
                col1, col2 = st.columns(2)
                with col1:
                    # Button Toggle ChatBot History
                    if st.button(
                        "Toggle ChatBot History",
                        use_container_width=True,
                        # disabled button (Streamlit 1.22+ supports disabled param)
                        disabled=st.session_state["running"],
                    ):
                        st.session_state.show_history = not st.session_state.get(
                            "show_history", True
                        )
                with col2:
                    # Button Toggle ChatBot History
                    if st.button(
                        "Clear ChatBot History",
                        use_container_width=True,
                        # disabled button (Streamlit 1.22+ supports disabled param)
                        disabled=st.session_state["running"],
                    ):
                        st.session_state.messages = []
            with chat_btn_placeholder:
                # React to user input
                # prompt := st.chat_input("What is up?")
                if prompt := st.chat_input(
                    "Ask Assistant: say something and/or attach an image",
                    accept_file=True,
                    file_type=["jpg", "jpeg", "png"],
                ):
                    st.session_state["running"] = True
                    if prompt and prompt.text:
                        query = prompt.text.strip()
                        # Add user message to chat history
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": query,
                                "type": "text",
                            }
                        )
                        # Replace this with your assistant logic
                        # response = f"Echo: {query}"
                        # Sens user's message to the LLM and get a response
                        # messages = [
                        #     {"role": "system", "content": "You are a helpful assistant"},
                        #     *st.session_state.messages ? tokenize
                        # ]
                        # assitant_response = client.chat.completions.create(
                        #     model = st.session_state["groqai_model"],
                        #     messages = messages
                        # )
                        # response = assitant_response.choices[0].message.content
                        # logger.info(response)
                        response = chat_provider.get_response(
                            st.session_state.messages,
                            api_key=key_input,
                        )
                        # Add assistant response to chat history
                        # Display assistant response in chat message container
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "type": "text",
                            }
                        )
                    elif prompt and prompt["files"]:
                        query = prompt["files"][0]
                        # Replace this with your assistant logic
                        response = query
                        # Add user message to chat history
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": query,
                                "type": "image",
                            }
                        )
                        # Add assistant response to chat history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "type": "image",
                            }
                        )
                    st.session_state["running"] = False
            with chat_history_placeholder:
                if st.session_state.get("show_history", True):
                    st.write("History is visible.")
                    with st.expander("ChatBot History", expanded=True):
                        # Display bordered chat container messages from history on app rerun
                        for message in st.session_state.messages:  # loop Q&A
                            # Display user message in chat message container
                            # Display assistant response in chat message container
                            # method 2
                            # st.chat_message("user").markdown(prompt)  # short form
                            # st.chat_message("user").image(query)  # short form
                            # method 1
                            with st.chat_message(message["role"]):
                                if message["type"] == "text":
                                    st.markdown(message["content"])
                                elif message["type"] == "image":
                                    st.image(message["content"])
                else:
                    pass
            with chat_messages_placeholder:
                if len(st.session_state.messages) >= 2:  # noqa: PLR2004
                    for message in st.session_state.messages[-2:]:  # loop Q&A
                        # Display user message in chat message container
                        # Display assistant response in chat message container
                        # st.chat_message("user").markdown(prompt)  # short form
                        # st.chat_message("user").image(query)  # short form
                        with st.chat_message(message["role"]):
                            if message["type"] == "text":
                                st.markdown(message["content"])
                            elif message["type"] == "image":
                                st.image(message["content"])
                else:
                    pass

    # ---------------------- Entrypoint ----------------------

    if __name__ == "__main__":
        run_chat_ui()
