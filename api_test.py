# En mode debugging (et non deploiement)
# To test my json api:
# - run this in a second instance of spyder
#   !! actually don't need a 2nd spyder, just a 2nd CLI!
#   after launching the streamlit server (local) on the 1st spyder kernel
#   (could also download the tool postman to test api)
# - open a new anaconda CLI + conda activate oc_p7(choose environment displayed
#   at bottom right in Spyder) + cd...(copy paste from adress at top of Spyder)
# Run streamlit in local, in another anaconda CLI:
#   streamlit run app_streamlit.py (will open the app in the browser)
# In both api and streamlit may need to write back the localhost urls


import requests

# res = requests.post('http://localhost:5000/predict',
#                     json={"id":400377})  # train
# res = requests.post('http://localhost:5000/predict',
#                     json={'id': 30000})  # test
res = requests.post('https://bank-app-oc.herokuapp.com//predict',
                    json={'id': 12500})  # test
if res.ok:
    print(res.json())
print(res.ok, res)
