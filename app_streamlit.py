# Streamlit app
#    To run, in a CLI:   streamlit run app_streamlit.py


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from PIL import Image
import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



CLOUD = True  # True: deployment on the cloud / False: local

st.set_option('deprecation.showPyplotGlobalUse', False)
# mpl.rcParams["font.size"] = 3



@st.cache   # put in memory for faster rendering at each update
def get_shap(model, train_df, test_df):
    """Compute SHAP values."""
    # # If compute shap inside the app (but slow):
    # explainer = shap.explainers.Tree(model)
    # shap_values = explainer.shap_values(test_df)
    # exp_shap_values = explainer(test_df)
    # shap_values_train = explainer.shap_values(train_df)

    # if load shap from pickle
    # shap_values, exp_shap_values, shap_values_train = \
    shap_values, exp_shap_values = \
        joblib.load('data/shap.jlb')
    return (shap_values, exp_shap_values)  # , shap_values_train
    # exp_shap_values = joblib.load('data/shap.jlb')
    # return exp_shap_values

# # @st.cache
# def plot_global_imp(shap_values_train, train_df):
#     """."""
#     g4 = shap.summary_plot(shap_values_train, train_df, plot_size=(12, 8),
#                            # show=False
#                            )
#     return g4  # , plt.gcf()


def calc_score(x, seuil):
    """Compute the score based on the 'proba' of class 1."""
    if x < seuil:
        res = 0.5 / seuil * x
    else:
        res = x * 0.5 / (1 - seuil) + (0.5 - seuil) / (1 - seuil)
    return int((1 - res) * 100)


def display_title():
    """."""
    st.markdown("<h1 style='text-align: center; color: blue;'>"
                "Prêt à Dépenser</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: black;'>"
                "Credit Scoring </h2>", unsafe_allow_html=True)
    st.write('') ; st.write('')


def display_results(res):
    """."""
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
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer(test_df)  # w/ cache?
    # shap_values, exp_shap_values, shap_values_train = \
    (shap_values, exp_shap_values) = \
        get_shap(model, train_df, test_df)
    # exp_shap_values = get_shap(model, train_df, test_df)

    # Waterfall plot

    # fig = plt.figure(figsize=(6, 2))
    # plt.figure(figsize=(0.1,0.1))
    g1 = shap.plots.waterfall(exp_shap_values[row_n], max_display=15)
    # fig = plt.gcf() ; fig.set_size_inches(2, 2) ; fig.set_dpi(2)
    # _, h = plt.gcf().get_size_inches()
    # plt.gcf().set_size_inches(h*5, h)
    st.pyplot(g1)
    st.write('') ; st.write('')


    # Distribution plots

    st.markdown('### Cross-comparison on 1 factor:')
    # st.write(shap_values[row_n].values)
    # feature_names2 = test_df.columns[
    #     np.argsort(np.abs(shap_values[row_n].values))]
    # feature_names = shap_values[row_n].feature_names  # ordered by shap?
    # st.write(feature_names[:5])
    # st.write(feature_names2[:5])

    feature_names = exp_shap_values.feature_names
    shap_df = pd.DataFrame(exp_shap_values.values, index=test_df.index,
                           columns=feature_names)  # if not lgbm

    # vals= np.round(np.abs(shap_df[row_n].values), 6)
    vals = np.round(np.abs(shap_df.loc[customer_id].values), 6)
    feat_imp = pd.DataFrame(list(zip(feature_names, vals)),
                            columns=['col_name', 'feature_importance_vals'])
    feat_imp.sort_values(by=['feature_importance_vals'], ascending=False,
                         inplace=True)
    # st.write(feat_imp[:10])
    # st.write(feat_imp['col_name'], feat_imp.iloc[0]['col_name'])
    # st.write(shap_df[:10])

    var_plot = st.multiselect('Choose the features to plot',
                              feat_imp['col_name'],
                              'EXT_SOURCE_1')
                              # feat_imp.iloc[0]['col_name'])
                              # feature_names, feature_names[0])
    full_train = pd.concat([train_df, Y_df], axis=1)

    fig = None
    for var in var_plot:
        fig, ax = plt.subplots(figsize=(3, 3))
        g2 = sns.kdeplot(data=full_train, x=var, hue='TARGET',
                         common_norm=False, ax=ax)
        val_cust = test_df.loc[customer_id, var]
        g2.axvline(x=val_cust, color='g', ls='--', label='cust')
        fig = plt.gcf()
        fig.set_size_inches(8, 3)  # ; fig.set_dpi(100)
        st.pyplot()  # fig)

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
                            # [feat_imp.iloc[0]['col_name'],
                            #  feat_imp.iloc[1]['col_name']],
    )
    # g3 = sns.jointplot(x=full_train[var_bi[0]],
    #                    y=full_train[var_bi[1]],
    #                    hue=full_train['TARGET'],
    #                    marker="s", s=20, palette='Set1',  # kind="kde"
    #                    height=8, ratio=8)
    # g3.ax_joint.scatter(x=test_df.loc[customer_id, var_bi[0]],
    #                     y=test_df.loc[customer_id, var_bi[1]],
    #                     marker='o', color='g', s=200)
    # g3.ax_joint.set_xlim(-0.01, 1.0) ; g.ax_joint.set_ylim(-0.01, 1.0)

    g3 = sns.JointGrid(data=full_train, x=var_bi[0], y=var_bi[1],
                       hue='TARGET',  # palette='Set1',  # kind="kde"
                       height=8, ratio=8,
                       # xlim=(-0.01, 1.0), ylim=(-0.01, 1.0)
                       )
    g3.plot_joint(sns.scatterplot, marker="s", s=20)
    g3.plot_marginals(sns.kdeplot, common_norm=False)
    g3.ax_joint.scatter(x=test_df.loc[customer_id, var_bi[0]],
                        y=test_df.loc[customer_id, var_bi[1]],
                        marker='o', color='g', s=200)

    # fig = plt.gcf() ; fig.set_size_inches(2,0.5) #; fig.set_dpi(500)
    st.pyplot(g3)  # , clear_figure=True)
    st.write('') ; st.write('')
    plt.clf()


    # Customer Data

    st.markdown('### Customer data:')
    st.dataframe(test_df.loc[customer_id].T)
    st.write('') ; st.write('')


    # Best global features for the model (based on permutation importance)

    st.markdown('### Influence of the top factors '
                'for the model (global):')
    # g4 = plot_global_imp(shap_values_train, train_df)
    # # plt.gcf().axes[-1].set_aspect(100)
    # # plt.gcf().axes[-1].set_box_aspect(100)
    # # fig = plt.gcf() ; fig.set_size_inches(2,0.5) #; fig.set_dpi(500)
    # st.pyplot(g4)
    # st.write('') ; st.write('')

    # Use image to speed up
    image_shap = Image.open('img/shap_global_importance.png')
    st.image(image_shap, width=900)  # , caption='', width=200)



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
                               test_df.index, index=8)

# _, col2c, _ = st.sidebar.columns([1,1,1])
# with col2c:
#     button_score = st.button('Get Score')
# if button_score:
#     # FIRST_LOAD = False
#     # display_title()
#     # res0 = requests.post('https://bank-app-oc.herokuapp.com//predict',
#     #                      json={'id':customer_id}) #test
#     res0 = requests.post('http://localhost:5000/predict',
#                          json={'id':customer_id})
#     res = res0.json()
#     # display_results(res)




# -- MAIN PAGE

display_title()

# Send a POST request to the API to getcompute the customer score
if CLOUD:
    res0 = requests.post('https://bank-app-oc.herokuapp.com//predict',
                         json={'id': customer_id})
else:
    res0 = requests.post('http://localhost:5000/predict',
                         json={'id': customer_id})
# print(res0.json())
display_results(res0.json())




# -- THE END --




# # if app_mode=='Home':
# # st.title('Employee Prediction')
# st.markdown('Data :')
# #df=pd.read_csv('emp_analytics.csv') #Read our data dataset
# st.dataframe(test_df.head())

# # Page predict
# elif app_mode == 'Predict_Churn':
# ## specify our inputs
#     st.subheader('Fill in employee details to get prediction ')
#     st.sidebar.header("Other details :")
#     prop = {'salary_low': 1, 'salary_high': 2, 'salary_medium': 3}
#     satisfaction_level = st.number_input("satisfaction_level", min_value=0.0,
#                                          max_value=1.0)

# for shap: st.pyplot(shap.plots.force(shaps_values[0],matplotlib=True))
