# Streamlit app

# Import modules

import streamlit as st
import pandas as pd
import numpy as np

# Run app
st.title('My first app')

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option
