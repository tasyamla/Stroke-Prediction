# import libraries
import streamlit as st
import numpy as np
import pandas as pd

# import modules
import EDA
import prediction


def main():
    PAGES = {
        "EDA": EDA,
        "Prediction": prediction
    }

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

# main idiom
if __name__ == "__main__":
    main()