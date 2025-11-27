import streamlit as st
import pandas as pd
import numpy as np

st.title("🚀 AI Sales Dashboard")
st.write("Initial Version - More features coming soon!")

# Basic functionality
data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Sales': [50000, 75000, 60000, 90000, 85000, 95000]
})

st.line_chart(data.set_index('Month'))
st.success("Live Dashboard Successfully Deployed! 🎉")
