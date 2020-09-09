library(httr)
library(jsonlite)

L1000FWD_URL <- '{{ config.ORIGIN }}{{ config.ENTER_POINT }}/'

result_id <- '5a01f822a5d0d538b1b7cb48'
response <- GET(paste0(L1000FWD_URL, 'result/topn/', result_id))
if (response$status_code == 200){
	response <- fromJSON(httr::content(response, 'text'))
	print(response)
}
