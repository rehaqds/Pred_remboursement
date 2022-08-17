# Define the API
#    Take a json (customer id) in and a json out (class and proba default)
#    Deployed on Heroku
#    Used by Streamlit app
#    Actually don't need this API as work could be done in Streamlit script
#       directly (but useful for practice/learn)

# To test locally: in one CLI: python api.py (& http://localhost:5000)
#    in a 2nd CLI: launch streamlit in local: streamlit run app_streamlit.py



import joblib
# import requests
# import json
import numpy as np
# import sklearn
# import xgboost
import catboost   # used by joblib
from flask import Flask, request, jsonify



THRESHOLD1 = 0.16  # 0.065


# Your API definition
app = Flask(__name__)
    # name= '__main__' if run directly. if imported: name of file
app.config["DEBUG"] = True
    # lance le débogueur, ce qui permet d’afficher un message autre que
    #    « Bad Gateway » s’il y a une erreur dans l’application


# Load the data
train_df = joblib.load('data/train_df.jlb')
model = joblib.load('data/p7-model.jlb')
test_df = joblib.load('data/test_df.jlb')
# with open('p7-model.pkl', 'rb') as f1:
#     model = pickle.load(f1)
print('Model loaded\n')



# The decorator associates a function to a url
# flask envoie des requêtes http à des fonctions (routage)
# Main page
@app.route('/')
def start():
    """."""
    return('<h1>P7 API running</h1>')
    # will be displayed on the root page: http://localhost:5000/
    #   or https://bank-app-oc.herokuapp.com/ if deployed to heroku



# Predict page
@app.route('/predict', methods=['POST'])
def predict():
    """."""
    json_ = request.json  # json received as input (customer ID)
    # json_ = json.loads(json_.de%tbcode("utf-8"))
    print(json_)
    # cust = test_df[test_df['SK_ID_CURR'] == json_['id']]
    cust = test_df.loc[[json_['id']]]
        # need a list to get a df size (53,1) and not a series size (53,)

    # As .predict use 0.5 threshold, needs to use predict_proba
    # prediction = model.predict(cust.values)  # don't forget .values !!!
    # [0, 1]: O for the first and only line in the array and 1 to get the
    # probability for class 1
    proba = model.predict_proba(cust.values)[0, 1]
    prediction = np.where(proba < THRESHOLD1, 0, 1)
    print(prediction, proba)

    return jsonify({'class': str(prediction),
                    'proba': str(proba),
                    'seuil': str(THRESHOLD1)
            })  # to json format (must be a string to be json serializable)



# lance le serveur (remove debug when not needed)
# adresse IP du serveur local: http://127.0.0.1:5000 (=http://localhost:5000)
if __name__ == '__main__':


    app.run()  # en phase production

    # app.run(debug=True, use_reloader=False)  # en phase prog #port=9010
    # use_reloader=False needed when debug=True if run from spyder/ipython
    #   if not run api.py from the CLI
