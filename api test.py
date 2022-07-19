# to test my json api, run this a second instance of spyder 
#   after launching the server on the 1st spyder kernel 
#   (could also download the tool postman)


import requests

# res = requests.post('http://localhost:5000/predict', json={"id":400377}) #train
res = requests.post('http://localhost:5000/predict', json={'id':30000}) #test
if res.ok:
    print(res.json())
