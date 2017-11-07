library(httr)
library(jsonlite)

L1000FWD_URL <- 'http://amp.pharm.mssm.edu/L1000FWD/'

sig_id <- 'CPC006_HA1E_24H:BRD-A70155556-001-04-4:40'
response <- GET(paste0(L1000FWD_URL, 'sig/', sig_id))
if (response$status_code == 200){
	response <- fromJSON(httr::content(response, 'text'))
	print(response)
}
