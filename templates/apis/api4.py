import json, requests
from pprint import pprint

L1000FWD_URL = '{{ config.ORIGIN }}{{ config.ENTER_POINT }}/'

result_id = '5a01f822a5d0d538b1b7cb48'
response = requests.get(L1000FWD_URL + 'result/topn/' + result_id)
if response.status_code == 200:
	pprint(response.json())
	json.dump(response.json(), open('api4_result.json', 'wb'), indent=4)
