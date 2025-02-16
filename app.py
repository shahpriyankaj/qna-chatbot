'''Streamlit frontend for interacting with the chatbot.'''
import streamlit as st
import requests
import json
import sqlite3

FASTAPI_URL = "http://127.0.0.1:8000/chat"

st.title("QnA Chatbot")
# User input for prompt
user_question = st.text_input("Enter your text:")
full_response = ""
if st.button("Send"):
    if user_question:
        with st.empty():  # Creates a placeholder for real-time streaming updates
            response = requests.post(FASTAPI_URL, json={"question": user_question}, stream=True)
            print(f"Response Status Code: {response.status_code}")
            full_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        decoded_chunk = chunk.decode("utf-8")

                        data = json.loads(decoded_chunk)  # Parse JSON response
                        if "response" in data:
                            full_response += data["response"] # Append to response
                            st.markdown(full_response)  # Update UI dynamically
                        else:
                            print("Warning: 'response' key missing in chunk.")
                            st.markdown("Warning: Unexpected Error occured. Please contact your administrator.")
                    except json.JSONDecodeError as e:
                        print(f"JSON Decode Error: {e}")
                        st.markdown("Warning: Unexpected Error occured. Please contact your administrator.")
                        pass
                    except Exception as e:
                        print(f"Unexpected Error: {e}")
                        st.markdown("Warning: Unexpected Error occured. Please contact your administrator.")
    
# Initialize session state for thumbs
if 'thumbs' not in st.session_state:
    st.session_state.thumbs = None
thumbs_up = st.button("👍 Thumbs Up")
thumbs_down = st.button("👎 Thumbs Down")

if thumbs_up:
    st.session_state.thumbs = "thumbs_up"
elif thumbs_down:
    st.session_state.thumbs = "thumbs_down"

# Connect to SQLite database to store the Thumbs Up and Thumbs Down for evaluation
conn = sqlite3.connect('qna_interactions.db')
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS interactions(question TEXT, response TEXT, thumbs TEXT)''')

# For this prototype, only storing the interaction when user is clicking on Thumbs Up or Thumbs Down button.
# Ideally, the database operations should be done in back-end. As well as, we can store all user interactions to store conversation history.
if st.session_state.thumbs:
    c.execute("INSERT INTO interactions (question, response, thumbs) VALUES (?, ?, ?)", 
                (user_question, full_response, st.session_state.thumbs))
    conn.commit()