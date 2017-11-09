library(sensitivity); library(tidyverse); library(dplyr)

setwd("~/GitHub/maine_infant_mortality_data")
infmort <- read_tsv("./data/us_infant_mortality_2.tsv")

OAT <- function(infmort, USstatecase, howmuch = 1){  # function(frame, case, howmuch, sort.cols = NULL, sort.asc = NULL){
  # if (is.null(sort.cols) = F )
  #   frame <- frame %>%
  #     arrange(frame, sort.cols, ...)
  #####
  #
  # bdn <- bdn %>%
  # gather(datecat, measure, 2:61, convert=T) %>%
  #   separate(datecat, c("year","infant"), sep="_") %>%
  #   spread(infant, measure) %>%
  #   rename(state = X1, births = Births, deaths = Deaths, imr = IMR)  
  #####
  #
  infmort <- infmort %>%
    gather(datencat, measure, 2:length(infmort)) %>%
    separate(datencat, c("year","infant"), sep = "_", convert = T) %>%
    spread(infant, measure) %>%
    rename(state = State, births = Births, deaths = Deaths)
  
  # create a list of the years with 'state' at the initial position
  ## R, make a df with state years as the UoA
#  out.rates <- data.frame(infmort$state, infmort$year)
  
  # ✔ make a list of 'rates'
  
  # ✔ states as list drawn from the 'index' index in the raw/infmort object/structure
  
  # For each state-year calculate 
    # birth value
      # get birth-value
      # b <- # for the state passed in, grab its raw births
    
    # death value
      #get death value
      # d <- # grab the raw deaths and add the 'howmuch' parameter
    # d = 0 if the d in the infmort set is <0 (so if stmt above)
 infmort %>%
          mutate(rate = (deaths + howmuch)/births) 
      # Error: Error in mutate_impl(.data, dots) : 
      # Evaluation error: non-numeric argument to binary operator.
      
  # calculate the value of d/b, then place it in a cell appropriate to that year
  # if more than one state, create a new row for each state (where columns represent years of the IMR)
#return a formatted dataframe containing the row, column of state-years
  
# rank the  numerical data ranks (1 through n) among the states (byt the states column) for each year; python's axis = 0 means rank by the column, while axis = 1 means rank by the row). "A 2-dimensional array has two corresponding axes: the first running vertically downwards across rows (axis 0), and the second running horizontally across columns (axis 1)." https://docs.scipy.org/doc/numpy-1.10.0/glossary.html
          
}

# 

raterank <- OAT(infmort = infmort, "Maine", howmuch = 1)
