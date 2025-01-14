import streamlit as st
import os
from time import sleep
from dotenv import load_dotenv

# Store the allowed users and passwords
USERS = {
    "roshan": "Trmeric@123",
    "user2": "password2",
    "paul": "Veolia@159"
}

load_dotenv()
print("--debug env loaded",os.getenv("OCR_AGENT"))

# Create the login function
def login():
    st.title("Welcome to trmeric evaluation service")
    st.write("Please log in to continue.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Check login credentials
    if st.button("Log in", type="primary"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
            sleep(0.5)  # Give some feedback time
        else:
            st.error("Incorrect username or password")

# Initialize session state for login if not done
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# If logged in, render app.py functionality
if st.session_state.logged_in:
    import app  # Importing app.py after successful login
    app.main()  # Call the main function from app.py
else:
    login()  # Show the login page if not logged in
