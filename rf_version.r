library(readr)
library(xgboost)

#my favorite seed^^

cat("reading the train and test data\n")
train <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")
store <- read_csv("data/store.csv")



# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)

# looking at only stores that were open in the train set
# may change this later
train <- train[ which(train$Open=='1'),]

# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
train <- train[,-c(3,8)]

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column (since elements are extracted) and also StateHoliday which has a lot of NAs (may add it back in later)
test <- test[,-c(4,7)]

feature.names <- names(train)[c(1,2,5:19)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
max(train$year)
min(train$year)
tra2<-train[train$year==14,]
tra1<-train[train$year==13,]

RMPSE<- function(preds, dtrain)
{
  labels <- getinfo(dtrain, "label")
  eratio<- (exp(as.numeric(preds))-1) / (exp(as.numeric(labels))-1)
  eratio[mapply(is.infinite, eratio)] <- 2
  err <- sqrt(mean((eratio-1)^2))
  return(list(metric = "RMPSE", value = err))
}

set.seed(123)

dval<-xgb.DMatrix(data=data.matrix(tra1[,feature.names]),label=log(tra1$Sales+1))
dtrain<-xgb.DMatrix(data=data.matrix(tra2[,feature.names]),label=log(tra2$Sales+1))
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.9, # 0.7
                colsample_bytree    = 0.7 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 300, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                   early.stop.round    = 10,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)

                