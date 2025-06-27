import streamlit as st # type: ignore
import runpy



# Store the previously selected option in session state
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

option = st.sidebar.radio("Choose Functionality:", ["Gesture Chatbot 🤖", "Gesture-to-System Commands 🔧"])

# If the selected option changes, clear the previous output and rerun
if option != st.session_state.selected_option:
    st.session_state.selected_option = option
    st.rerun()

if option == "Gesture Chatbot 🤖":
    st.sidebar.write("Running Gesture Chatbot...")
    runpy.run_path("chatbot.py")  # Run chatbot script within the same process
elif option == "Gesture-to-System Commands 🔧":
    st.sidebar.write("Running Gesture-to-System Commands...")
    runpy.run_path("systemcommands2.py")  # Run system commands script within the same process
