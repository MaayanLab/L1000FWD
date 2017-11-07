import json, requests
from pprint import pprint

L1000FWD_URL = 'http://amp.pharm.mssm.edu/L1000FWD/'

sig_id = 'CPC006_HA1E_24H:BRD-A70155556-001-04-4:40'
response = requests.get(L1000FWD_URL + 'sig/' + sig_id)
if response.status_code == 200:
	pprint(response.json())
	json.dump(response.json(), open('api2_result.json', 'wb'), indent=4)
