import streamlit as st

st.title("Lab 3")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.set_page_config(page_title="Multi-Page Labs", page_icon="🧪")

lab2 = st.Page("lab2.py", title="Lab 2", icon="🧪")
lab1 = st.Page("lab1.py", title="Lab 1", icon="📄")
lab3 = st.Page("lab3.py", title="Lab 3", icon="💬")
lab4 = st.Page("lab4.py", title="Lab 4", icon="💬")
lab5 = st.Page("lab5.py", title="Lab 5", icon="💬")

nav = st.navigation(pages=[lab2, lab1, lab3, lab4, lab5])   
nav.run()
