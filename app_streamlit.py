# Streamlit app
# To run, in a CLI:   streamlit run app_streamlit.py
# 12/10/2022: commented out blocks of code removed


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from PIL import Image
import shap
import matplotlib.pyplot as plt
import seaborn as sns



CLOUD = True  # True: deployment on the cloud / False: local

st.set_option('deprecation.showPyplotGlobalUse', False)



@st.cache   # put in memory for faster rendering at each update
def get_shap(model, train_df, test_df):
    """
    Compute SHAP values.
    To save time, SHAP values are pre-computed and saved in a file
    """
    shap_values, exp_shap_values = joblib.load('data/shap.jlb')
    return (shap_values, exp_shap_values)



def calc_score(x, seuil):
    """
    Compute the score based on the 'proba' of class 1.
    bilinear function with a break at x=seuil where score=0.5 ie 50%
    """
    if x < seuil:
        res = 0.5 / seuil * x
    else:
        res = x * 0.5 / (1 - seuil) + (0.5 - seuil) / (1 - seuil)
    return int((1 - res) * 100)


def display_title():
    """Display the title."""
    st.markdown("<h1 style='text-align: center; color: blue;'>"
                "Prêt à Dépenser</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>"
                "Credit Scoring </h2>", unsafe_allow_html=True)
    st.write('') ; st.write('')


def display_results(res):
    """Display all the score and the plots."""
    # Score
    score = calc_score(float(res['proba']), float(res['seuil']))
    if int(res['class']) == 0:
        st.success('LOAN APPROVED')
        st.write('')
        st.metric(label="Sccore: ",
                  value=str(score) + '/100',
                  delta="Approved")
    elif int(res['class']) == 1:
        st.error('LOAN REJECTED')
        st.write('')
        st.metric(label="Score: ",
                  value=str(score) + '/100',
                  delta="-Rejected")
    st.write('(', round(float(res['proba']), 2), ')')
    st.write('') ; st.write('')


    # SHAP plots
    st.markdown('### Influence of the top factors concerning the decision '
                'for the customer:')
    row_n = test_df.index.get_loc(customer_id)  # get line n°
    (shap_values, exp_shap_values) = get_shap(model, train_df, test_df)


    # Waterfall plot
    g1 = shap.plots.waterfall(exp_shap_values[row_n], max_display=15)
    st.pyplot(g1)
    st.write('') ; st.write('')


    # Distribution plots
    st.markdown('### Cross-comparison on 1 factor:')
    feature_names = exp_shap_values.feature_names
    shap_df = pd.DataFrame(exp_shap_values.values,
                           index=test_df.index,
                           columns=feature_names
    )  # if not lgbm
    vals = np.round(np.abs(shap_df.loc[customer_id].values), 6)
    feat_imp = pd.DataFrame(list(zip(feature_names, vals)),
                            columns=['col_name', 'feature_importance_vals']
    )
    feat_imp.sort_values(by=['feature_importance_vals'],
                         ascending=False,
                         inplace=True
    )
    var_plot = st.multiselect('Choose the features to plot',
                              feat_imp['col_name'],
                              'EXT_SOURCE_1'
    )
    full_train = pd.concat([train_df, Y_df], axis=1)

    fig = None
    for var in var_plot:
        fig, ax = plt.subplots(figsize=(3, 3))
        g2 = sns.kdeplot(data=full_train,
                         x=var,
                         hue='TARGET',
                         common_norm=False, ax=ax
        )
        val_cust = test_df.loc[customer_id, var]
        g2.axvline(x=val_cust, color='g', ls='--', label='cust')
        fig = plt.gcf()
        fig.set_size_inches(8, 3)
        st.pyplot()

    st.write('')
    st.write('Target 0 (blue): clients with no payment difficulties')
    st.write('Target 1 (orange): clients with payment difficulties: '
             'late payment more than X days on at least one of '
             'the first Y installments of the '
             'loan in our sample')
    st.write('Green dashed line : client')
    st.write('') ; st.write('')


    # Bi-variate plot
    st.markdown('### Cross-comparison on 2 factors:')
    var_bi = st.multiselect('Choose the 2 factors to plot',
                            feat_imp['col_name'],
                            ['EXT_SOURCE_1', 'EXT_SOURCE_2'],
    )
    g3 = sns.JointGrid(data=full_train,
                       x=var_bi[0],
                       y=var_bi[1],
                       hue='TARGET',
                       height=8,
                       ratio=8,
    )
    g3.plot_joint(sns.scatterplot, marker="s", s=20)
    g3.plot_marginals(sns.kdeplot, common_norm=False)
    g3.ax_joint.scatter(x=test_df.loc[customer_id, var_bi[0]],
                        y=test_df.loc[customer_id, var_bi[1]],
                        marker='o',
                        color='g',
                        s=200,
    )
    st.pyplot(g3)
    st.write('') ; st.write('')
    plt.clf()


    # Customer Data
    st.markdown('### Customer data:')
    st.dataframe(test_df.loc[customer_id].T)
    st.write('') ; st.write('')


    # Best global features for the model (based on permutation importance)
    # Use an image to speed up display
    st.markdown('### Influence of the top factors '
                'for the model (global):')
    image_shap = Image.open('img/shap_global_importance.png')
    st.image(image_shap, width=900)



# Load the data
model = joblib.load('data/p7-model.jlb')
test_df = joblib.load('data/test_df.jlb')
train_df = joblib.load('data/train_df.jlb')
Y_df = joblib.load('data/Y_df.jlb')



# -- SIDE BAR

image = Image.open('img/logo.png')
col1, col2, col3 = st.sidebar.columns([1, 2, 5])  # to center the image
with col2:
    st.image(image, caption='', width=200)  # must remove sidebar inside 'with'
_, col2b, _ = st.sidebar.columns([0.5, 1, 0.5])  # to center the selectbox
with col2b:
    customer_id = st.selectbox('Select the customer ID',
                               test_df.index,
                               index=8
    )



# -- MAIN PAGE

display_title()

# Send a POST request to the API to compute the customer score
if CLOUD:
    res0 = requests.post('https://bank-app-oc.herokuapp.com//predict',
                         json={'id': customer_id}
    )
else:
    res0 = requests.post('http://localhost:5000/predict',
                         json={'id': customer_id}
    )
display_results(res0.json())
