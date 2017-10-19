library(sensitivity); library(tidyverse); library(dplyr)

infmort <- read.delim("./data/us_infant_mortality.tsv", check.names = F)

cleanIMR <- function(frame, case, howmuch, sort.cols = NULL, sort.asc = NULL){
  if (is.null(sort.cols) = F )
    frame <- arrange(frame, sort.cols, ...)
  
}
# bdn <- bdn %>%
# gather(datecat, measure, 2:61, convert=T) %>%
#   separate(datecat, c("year","infant"), sep="_") %>%
#   spread(infant, measure) %>%
#   rename(state = X1, births = Births, deaths = Deaths, imr = IMR)


OAT <- function(cleanedDF){
  
}