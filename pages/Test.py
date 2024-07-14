import streamlit as st
from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "give me the script"}],
)
st.write(response.choices[0].message.content)
