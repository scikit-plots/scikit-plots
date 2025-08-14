# template_st_login.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=import-outside-toplevel
# pylint: disable=broad-exception-caught

"""template_st_login."""

# from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport

__all__ = []

# import streamlit as st
st = LazyImport("streamlit", package="streamlit")

# Use st.cache_data for immutable data and st.cache_resource for reusable, expensive resources
# Use @st.fragment to create modular, reusable UI blocks with proper state handling
if st:
    __all__ += [
        "st_login",
    ]

    # ---------------------- Streamlit Module Interface ----------------------
    def authenticate_user(username: str, password: str) -> bool:
        """Stub auth logic — replace with real DB or API."""
        return username in ["", "admin", "guest"] and password in ["", "admin", "guest"]

    def st_login() -> "tuple[bool, str]":
        """
        Render a login interface with both username/password and guest login.

        Returns
        -------
        Tuple (authenticated: bool, user_type: str)
            - authenticated: True if logged in
            - user_type: 'admin' or 'guest'
        """
        ## Initialize session state with defaults (only once)
        # st.session_state.setdefault("authenticated", False)
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = True  # 🔑 Always Login
            st.session_state.username = "guest"
            st.session_state.user_type = "guest"

        # Placeholder
        # st.container A static layout block.
        # st.empty().container Dynamic and replaceable container.
        login_placeholder = st.container()
        # Placeholder
        sidebar_placeholder = st.container()

        # Display login form only if not logged in
        if not st.session_state.authenticated:
            with login_placeholder:  # noqa: SIM117
                # st.title("🔐 Login Required")
                st.subheader("🔐 Login Required")
                tab1, tab2 = st.tabs(["🔑 Login", "👤 Guest"])

                # Tab 1: Admin Login
                with tab1:  # noqa: SIM117
                    with st.form("login_form"):
                        username = st.text_input("Username", key="guest").strip()
                        password = st.text_input("Password", type="password")
                        submitted = st.form_submit_button("Login")

                        if submitted:
                            # Replace with your own auth logic
                            if authenticate_user(username, password):
                                st.session_state.authenticated = True
                                st.session_state.username = username or "admin"
                                st.session_state.user_type = "admin"
                                st.success(
                                    f"👤 {username}, Logged in as admin successfully!"
                                )
                                st.rerun()  # rerun to load main app
                            else:
                                # st.error("Invalid username or password")
                                st.error("❌ Invalid credentials.")

                # Tab 2: Guest Login
                with tab2:
                    if st.button(
                        "Continue as Guest",
                        key="Continue as Guest",
                    ):
                        st.session_state.authenticated = True
                        st.session_state.username = username or "guest"
                        st.session_state.user_type = "guest"
                        st.info("You are logged in as Guest.")
                        st.rerun()
        else:
            # Clear login form
            login_placeholder.empty()

        with sidebar_placeholder:
            ## App starts below only after login
            authenticated = st.session_state.get("authenticated", False)
            username = st.session_state.get("username", "guest")
            user_type = st.session_state.get("user_type", "guest")

            ## After successful login
            if authenticated:
                ## sidebar UI
                st.sidebar.success(f"👋 Welcome, {username}!")
                if st.sidebar.button(
                    "🔓 Logout", key="🔓 Logout", use_container_width=True
                ):
                    for key in ["authenticated", "username", "user_type"]:
                        st.session_state.pop(key, None)
                    st.session_state.authenticated = False  # 🔑 Skip Always Login
                    st.rerun()
                ## main UI
                if user_type == "admin":
                    # st.title(f"🔑 {user_type.title()} dashboard loaded...")
                    st.subheader(f"🔑 {str(user_type).title()} dashboard loaded:")
                    # st.write(f"🔑 {user_type.title()} dashboard loaded...")
                else:
                    # st.title(f"{user_type.title()} session: limited access.")
                    st.subheader("Dashboard loaded:")
                    # st.write(f"{user_type.title()} session: limited access.")
        return st.session_state.get("authenticated", False)

    # ---------------------- Main Entrypoint ----------------------

    if __name__ == "__main__":
        st_login()
