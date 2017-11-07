library(httr)
library(jsonlite)

L1000FWD_URL <- 'http://amp.pharm.mssm.edu/L1000FWD/'

query_string <- 'dex'
response <- GET(paste0(L1000FWD_URL, 'synonyms/', query_string))
if (response$status_code == 200){
	response <- fromJSON(httr::content(response, 'text'))
	print(response)
}
