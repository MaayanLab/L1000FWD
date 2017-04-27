'''
To test if the Rook server is running properly.
'''
import os
import json
import requests

RURL = os.environ['RURL']

config = {"direction":"mimic","combination":False,"method":"geneSet"}
headers = {'content-type':'application/json'}
data = {
	"upGenes": ["ZNF238","ACACA","ACAT2","ACLY","ACSL3","C10ORF10","C14ORF1","CCL2","CCNG2","CD46","CDKN1A","CETN2","CLIC4","CYB5A","CYP1B1","CYP51A1","DBI","DDIT4","DHCR24","DHCR7","DSC3","DSG3","EBP","EFNA1","ELOVL5","ELOVL6","FABP7","FADS1","FADS2","FDFT1","FDPS","FGFBP1","FN1","GLUL","HMGCR","HMGCS1","HOPX","HS3ST2","HSD17B7","IDI1","IL32","INSIG1","IRS2","KHDRBS3","KRT14","KRT15","KRT6B","LDLR","LPIN1","LSS","MAP7","ME1","MSMO1","MTSS1","NFKBIA","NOV","NPC1","NSDHL","PANK3","PGD","PLA2G2A","PNRC1","PPL","PRKCH","PSAP","RDH11","SC5DL","SCD","SCEL","SDPR","SEPP1","SLC2A6","SLC31A2","SLC39A6","SMPDL3A","SNCA","SPRR1B","SQLE","SREBF1","SREBF2","STXBP1","TM7SF2","TNFAIP3","VGLL4","ZFAND5","ZNF185"],
	"dnGenes":["ARMCX2","BST2","CA2","F12","GDF15","GPX7","IFI6","IGFBP7","KCNJ16","KRT7","MDK","NID2","NOP56","NR2F6","PDGFRA","PROM1","RRP8","SAMSN1","SCG5","SERPINE2","SLC4A4","TRIP6"]
}

payload = dict(data.items() + config.items())
response = requests.post(RURL, data=json.dumps(payload),headers=headers)

if response.status_code == 200:
	print 'Success!'
	print response.json()['sig_ids'][:5]
	print response.json()['scores'][:5]

else:
	print 'Failed with status code:', response.status_code
