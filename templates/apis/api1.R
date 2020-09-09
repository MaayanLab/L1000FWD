library(httr)
library(jsonlite)

L1000FWD_URL <- '{{ config.ORIGIN }}{{ config.ENTER_POINT }}/'

query_string <- 'dex'
response <- GET(paste0(L1000FWD_URL, 'synonyms/', query_string))
if (response$status_code == 200){
	response <- fromJSON(httr::content(response, 'text'))
	print(response)
}
