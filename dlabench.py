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

st.set_page_config(layout="wide")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(0,50,100);
    textColor: rgb(0,255,255);
}
</style>""", unsafe_allow_html=True)

verbose = 0

st.title('Micron Deep Learning Accelerator (MDLA)')
st.write('Evaluation platform for MDLA Gen5 inference solutions')

model.modelManager(verbose)


####################################################
