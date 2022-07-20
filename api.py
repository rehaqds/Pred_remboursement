from flask import Flask, request, jsonify
import json 
import pickle, joblib
#import requests
#import pandas as pd
import sklearn
import xgboost


# Your API definition
app = Flask(__name__)
    # name= '__main__' if run directly. if imported: name of file
app.config["DEBUG"] = True
    # lance le débogueur, ce qui permet d’afficher un message autre que 
    #    « Bad Gateway » s’il y a une erreur dans l’application


# with open('p7-model.pkl', 'rb') as f1:
#     model = pickle.load(f1)
with open('train_df.pkl', 'rb') as f2:
    train_df = pickle.load(f2)  
model = joblib.load('p7-model.jlb')  
test_df = joblib.load('test_df.jlb')

print ('Model loaded\n')



# the decorator associates a function to a url
# flask envoie des requêtes http à des fonctions (routage)
@app.route('/')
def start():
    return('<h1>P7 API running</h1>')
    # will be displayed on the root page: http://localhost:5000/
    #   or https://bank-app-oc.herokuapp.com/ if deployed to heroku


@app.route('/predict', methods=['POST'])
def predict():
    print(2222)
    json_ = request.json
    # json_ = json.loads(json_.decode("utf-8"))
    print(json_)
    #cust = test_df[test_df['SK_ID_CURR'] == json_['id']]
    ###cust = test_df.loc[[json_['id']]]
    #cust = train_df.loc[[json_['id']]]
    # cust = train_df.iloc[[0]]   
        # need a list to get a df and not a series size (53,)

    prediction = model.predict(cust.values)  #don't forget .values !!!
    print(prediction)

    return jsonify({'score': str(prediction)})  #to json format




# lance le serveur (remove debug when not needed) 
# adresse IP du serveur local: http://127.0.0.1:5000 (=http://localhost:5000)
if __name__ == '__main__':


    app.run() # en phase production
    #app.run(debug=True, use_reloader=False)  # en phase prog #port=9010
    # use_reloader=False needed when debug=True if run from spyder/ipython
    #   if not run api.py from the CLI
    
    