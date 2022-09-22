"""
DLA bench
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import subprocess
import sys
import os
import model
from PIL import Image
from pathlib import Path
import base64
from datetime import datetime

verbose = 0

#------------

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

#------------

st.set_page_config(layout="wide")      #make app wide-screen

st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("sampleheader.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)

#sidebar-------------------------------------------
# Collect email recepient for session summary
email = st.sidebar.text_input(
    "Send session summary report to:",
    "Type registered email address here"
)

if st.sidebar.button("Start Session"):
    now = datetime.now()
    print(now)
    session_start = now.strftime("%d/%m/%Y %H:%M:%S")
    st.sidebar.write(session_start)
#sidebar-------------------------------------------

st.write("Choose from the available deployment configurations:")

model.modelManager(verbose)


####################################################
