library(lubridate)
library(gridExtra)

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

wd <- getwd()    
setwd(file.path(wd,c("Projects","movielens","rda")))
save(edx, validation, file="initial-set.rda")

# FIN STEP 0

# --------------------------
# Quiz: MovieLens Dataset
nrow(edx)
ncol(edx)
sum(edx$rating==0)
sum(edx$rating==3)
edx %>% select(movieId) %>% distinct() %>% nrow() # 10677 different movies
edx %>% select(userId) %>% distinct() %>% nrow() # 69878 different users
# How many movie ratings are in each of the following genres in the edx dataset?
edx %>% filter(str_detect(genres, "Drama" )) %>% nrow # 3910127
edx %>% filter(str_detect(genres, "Comedy" )) %>% nrow # 3540930
edx %>% filter(str_detect(genres, "Thriller" )) %>% nrow # 2325899
edx %>% filter(str_detect(genres, "Romance" )) %>% nrow # 1712100

# # str_detect
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# Which movie has the greatest number of ratings?
edx %>% group_by(movieId,title) %>% summarize(n=n()) %>% arrange(desc(n))

# What are the five most given ratings in order from most to least?
edx %>% group_by(rating) %>% summarize(n=n()) %>% arrange(desc(n))

# True or False: In general, half star ratings are less common than whole star ratings (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
mean(edx$rating == round(edx$rating)) # 79% whole star ratings

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()


##############################
################################ 
## --------------------------
##
# function that computes the RMSE for vectors of ratings and their corresponding predictors


options(digits=7)

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



# --------------------------

edx %>% head()

#################################
## timestamp effect
## as a first step we are checking for a time effect in ratings, then we'll compute a corrective term to account for this in the following steps.

mu <- mean(edx$rating)


## rating evolution through time

## we consider the time effect with a one week granularity, as in the course material.
## we compute time effect as : difference between the current mean and the long term mean which is mu)

week_effect <- edx %>% mutate (date = as_datetime(timestamp), week=round_date(date,unit="week")) %>% 
  group_by(week) %>% 
  summarize(avgweekeffect = mean(rating)-mu) 

# we fit a loess model in order to smooth the time effect. We set the span to 5 years
span <- 5*52/nrow(week_effect)
weekfit<-loess(avgweekeffect~as.numeric(week),degree = 1, span = span, data = week_effect)

# plot showing the average rating each week (black points), with the associated geom_smooth in blue, and our loess time effect in red
# red and blue lines are very close, which we wanted.
edx %>% mutate (date = as_datetime(timestamp), week=round_date(date,unit="week")) %>%
  group_by(week) %>% 
  summarize(weekrating = mean(rating)) %>%
  mutate(timeeffect = mu+weekfit$fitted) %>%
  ggplot(aes(week, weekrating)) +
  geom_point() +
  geom_smooth() +
  geom_line(aes(week, timeeffect), color="red")

time_effect <- data.frame(week=week_effect$week, time_effect = weekfit$fitted)

edx <- edx %>% mutate (date = as_datetime(timestamp), week=round_date(date,unit="week")) %>% 
  left_join(time_effect) 

#
# rmse with only mean effect mu
predicted_ratings <- rep(mu,length(validation$rating))
rmse_val <- RMSE(predicted_ratings, validation$rating) # 1.061202
rmse_results <- tibble(method = "Prediction with the mean rating", RMSE = rmse_val)

#
# rmse with mu and time effect
# for time_effect and mu we keep the test set values (hopefully there is no gap in weeks)
predicted_ratings <- validation %>% 
  mutate (date = as_datetime(timestamp), week=round_date(date,unit="week")) %>% 
  left_join(time_effect, by="week") %>%
  mutate(pred = mu + time_effect ) %>% 
  pull(pred)

rmse_val <- RMSE(predicted_ratings, validation$rating) # 1.06007
rmse_results <- bind_rows(rmse_results, data_frame(method="Additional Time effect", RMSE = rmse_val ))
rmse_results %>% knitr::kable()


#################################
# model :  Y(u,i) = mu + TE(t) + b_u + b_i + epsilon(u,i)
# Y(u,i) is the rating of movie i by user i
# TE : time effect
# b_u : user effect
# b_i : movie effect
# epsilon(u,i) : independant random variables of mean 0

# we use a regularization factor lambda to minimize the total variance V = sum((y_hat-Y)^2)+lambda*(sum(b_i^2)+sum(b_u^2))

# 1/ we partition the train set edx in order to look for optimal lambda 
edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_lambda <- edx[-edx_index,]
temp <- edx[edx_index,]
# we make sure the test set does not contain unknown items 
test_lambda <- temp %>% 
  semi_join(train_lambda, by = "movieId") %>%
  semi_join(train_lambda, by = "userId")
removed <- anti_join(temp, test_lambda)
train_lambda <- rbind(train_lambda, removed)

# 2/ we look for lambda with lower rmse
lambdas <- seq(0, 30, 5)
rmses <- sapply(lambdas, function(l){
  b_i <- train_lambda %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - time_effect)/(n()+l)) 
  b_u <- train_lambda %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu - time_effect)/(n()+l))
  predicted_ratings <- 
    test_lambda %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + time_effect + b_i + b_u) %>% 
    pull(pred)
  return(RMSE(predicted_ratings, test_lambda$rating))
})

rmses
qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]

rm(train_lambda,test_lambda,temp, removed, edx_index)

#################################
# --------------------------
# we'll keep a value of 5 for regularization
lambda <-5

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu - time_effect)/(n()+lambda)) 
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu - time_effect)/(n()+lambda))


##
## assess our  RMSE against the validation set
# b_i and b_u are NOT re-computed using validation set 
# 

predicted_ratings <- validation %>% 
  mutate (date = as_datetime(timestamp), week=round_date(date,unit="week")) %>% 
  left_join(time_effect, by="week") %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + time_effect + b_i + b_u) %>% 
  pull(pred)

rmse_val <- RMSE(predicted_ratings, validation$rating)
rmse_val
# 0.8642395

rmse_results <- bind_rows(rmse_results, data_frame(method="User and movie effects model", RMSE = rmse_val ))
rmse_results %>% knitr::kable()

# FIN STEP 1



#################################
##
## explore the residuals


# look at the 100 users giving the  worst predictions
users_worst <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(residual = mu + b_i + b_u + time_effect - rating) %>%
  arrange(desc(abs(residual))) %>% 
  select(userId)  %>%  unique %>% slice(1:100) %>% pull(userId)

length(users_worst)

# look at 100 users randomly chosen
users_rand <- edx %>% 
  filter( userId %in% sample(userId, 100) ) %>% 
  pull(userId) %>% unique

length(users_rand)

# Remark: the average ratings for worst group is generally far from the global average (3.5), compared with the random set :
edx %>% filter(userId %in% users_worst) %>% summarise(mean(rating),sd(rating))
edx %>% filter(userId %in% users_rand) %>%  summarise(mean(rating),sd(rating))


#################################
# break by sub-genres 
#

# get the whole list of atomic values for column genres
genres_l <- edx %>% 
  pull(genres) %>% unique() 
genres_l <- str_split(genres_l,"\\|", simplify =  FALSE)
genres_l <- sapply(genres_l, function(x) {paste(x, sep=" ",collapse=" ") })
genres_l <- paste(genres_l, collapse=" ")
genres_l <- data.frame(str_split(genres_l," "))%>% unique 
names(genres_l)<-c("genre")
genre_columns <- str_c("genre",seq(1,nrow(genres_l)),sep="_")

## we look at the ratings broken by genre, for the two set 

edx_worst <- edx %>% filter(userId %in% users_worst) %>% 
  separate(genres,sep="\\|", into = genre_columns, fill="right",extra="drop") %>% 
  gather(dummy, genre, genre_columns) %>% select(-dummy) %>% filter(!is.na(genre))

edx_rand <- edx %>% filter(userId %in% users_rand) %>% 
  separate(genres,sep="\\|", into = genre_columns, fill="right",extra="drop") %>% 
  gather(dummy, genre, genre_columns) %>% select(-dummy) %>% filter(!is.na(genre))

edx_worst %>% group_by(genre) %>% summarize(count=n(), mean=mean(rating), median=median(rating),sd=sd(rating)) %>% arrange (count)
edx_rand %>% group_by(genre) %>% summarize(count=n(), mean=mean(rating), median=median(rating),sd=sd(rating)) %>% arrange (count)

## histograms of ratings for random selection of users, broken by genre
## this shows a general genre effect 
edx_rand %>% 
  group_by(genre, userId) %>% summarize(rating=mean(rating)) %>% 
  ggplot(aes(rating)) + geom_histogram() + facet_wrap(. ~ genre)


h1 <- edx_worst  %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(residual = mu + b_i + b_u + time_effect - rating) %>%
  ggplot(aes(x=genre,y=residual))+geom_boxplot()+ggtitle("100 worst residuals users")
h2 <- edx_rand  %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(residual = mu + b_i + b_u + time_effect - rating) %>%
  ggplot(aes(x=genre,y=residual))+geom_boxplot()+ggtitle("random sample 100 users")
grid.arrange(h1, h2, ncol = 2) # TODO scale

# => The genre effect is much more important in the group of worst rating users

## Interesting to note, if we compare the two subset we see the user effect is more important on the group "worst" 
# this shows worst group has a larger user effect, which could explain the amplification of residuals
h1 <- edx_worst %>% 
  left_join(b_u) %>% 
  group_by(userId) %>% 
  summarize(b_u = first(b_u) ) %>% 
  ggplot(aes(b_u)) + geom_histogram() + ggtitle("100 worst residuals users") + ylim(0,20)
h2 <- edx_rand %>% 
  left_join(b_u) %>% 
  group_by(userId) %>% 
  summarize(b_u = first(b_u) ) %>% 
  ggplot(aes(b_u)) + geom_histogram() + ggtitle("random sample 100 users") + ylim(0,20)
grid.arrange(h1, h2, ncol = 2)

## lines connect the ratings for each user among the different genres
edx_rand %>% group_by(userId, genre) %>% summarize(rating=mean(rating)) %>% 
  ggplot(aes(genre, rating, group=userId)) + geom_line(aes(color=factor(userId)), show.legend = FALSE)

edx_worst %>% group_by(userId, genre) %>% summarize(rating=mean(rating)) %>% 
  ggplot(aes(genre, rating, group=userId)) + geom_line(aes(color=factor(userId)), show.legend = FALSE)

# => these graphs show that for the set of worst users the variability between genres is much more important than in the random set.

rm (rand_users, worst_users, edx_rand,  edx_worst, h1, h2) 


# In conclusion, there seems to be a genre effect. That effect could be approximately similar for a large part of users. 
# But for users with the worst predictions the genre effect seems to be different from one user to another.
# So we should try take into account a variability of genre for each user

# --------------------------- 
#
## Model accounting for a mean rating per genre for each user
#
# y(i,u) = mu + TE_t + b_i + b_u + mean ( b_u_k )
# where:
# b_u_k : average residual for user u and movies of genre k
# the mean is calculated over the genres k related to movie i  

# first modify edx to tidy the composite genres column to atomic genres 
# create a movie/genre separate table to optimize memory consumption

moviegenres <- edx %>% select(movieId,genres) %>% distinct() %>%
  separate(genres,sep="\\|", 
           into = tidyselect::all_of(genre_columns), 
           fill="right",
           extra="drop") %>% 
  gather(dummy, genre, tidyselect::all_of(genre_columns)) %>% select(-dummy) %>% filter(!is.na(genre)) 

head(moviegenres)

# now evaluate the new predictor. Nb. regularisation here seems to bring no benefit.

b_u_k <- edx %>% 
  left_join(moviegenres, by="movieId") %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(residual = mu + b_i + b_u + time_effect - rating) %>%
  group_by(userId, genre) %>%
  summarize(b_u_k = sum(rating -time_effect - b_i - b_u -mu)/n()) 



#
# Compute RMSE on training set
predicted_train <- edx %>% 
  left_join(moviegenres, by="movieId") %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_u_k, by = c("userId","genre")) %>%  
  group_by(userId,movieId) %>%
  summarize(b_uki = mean(b_u_k), b_u = first(b_u), b_i = first(b_i), rating = first(rating), time_effect = first(time_effect)) %>% 
  mutate( estimate = mu + time_effect + b_i + b_u + b_uki, pred = ifelse(estimate > 5,5, estimate))%>% 
  select(userId,movieId,pred)

predicted_ratings_train <- left_join(edx, predicted_train , by=c("userId","movieId")) %>% pull(pred)

# example: user id 1 with movie id 122
edx %>% filter(userId==1 & movieId==122) %>% knitr::kable()
ex_bi<-b_i %>% filter(movieId==122) %>% pull(b_i)
ex_bu<-b_u %>% filter(userId==1) %>% pull(b_u)
ex_te <- edx %>% filter(userId==1 & movieId==122) %>% pull(time_effect)
ex_buk <- b_u_k %>% filter(userId==1 & genre %in% c("Comedy","Romance"))
# The user-genre contribution shows 2 lines for the considered movie
ex_buk %>% knitr::kable()
# we have a contribution of mean(ex_buk$b_u_k) for the genre effect
# The final prediction is  
tibble (mu=mu, b_u=ex_bu, b_i=ex_bi,b_uki=mean(ex_buk$b_u_k), te=ex_te) %>% mutate(prediction = mu + b_u + b_i + te + b_uki)

RMSE(predicted_ratings_train, edx$rating) 
# 0.8091308 

# Compute RMSE on validation set
# Note that it can happen that a combination of user /movie was not present in the training set. 
# It is also possible that we miss some user-genre combinations. In that case the effect will be null for the given genre.  

pred_val <- validation %>%  
  mutate (date = as_datetime(timestamp), week=round_date(date,unit="week")) %>% 
  left_join(time_effect, by="week") %>%
  left_join(moviegenres, by = "movieId") %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_u_k, by = c("userId","genre")) %>%
  group_by(userId,movieId) %>%
  summarize(b_uki = mean(b_u_k, na.rm = TRUE), 
            b_u = first(b_u), 
            b_i = first(b_i), 
            rating = first(rating),
            time_effect = first(time_effect)) %>%
  mutate( estimate = mu + time_effect + b_i + b_u + ifelse(!is.na(b_uki), b_uki,0), 
          pred = ifelse(estimate > 5,5, ifelse(estimate<1,1,estimate))) %>% 
  select(userId,movieId,pred)

# any(is.na(predicted_ratings$pred))
# predicted_ratings%>%filter(is.na(pred))

predicted_val <- left_join(validation, pred_val , by=c("userId","movieId")) %>% pull(pred)
rmse_val <- RMSE(predicted_val, validation$rating) 
# 0.8489828 avec b_i et b_u
# 0.8235151 avec val_b*
rmse_results <- bind_rows(rmse_results, data_frame(method="Additional User-Genre effect model", RMSE = rmse_val ))
rmse_results %>% knitr::kable()

#rm(pred_val)

###
### matrix factorisation
### 

# explore
edx %>% filter(movieId %in% sample(movieId,50) & userId %in% sample(userId,50)) %>%
  left_join(predicted_train , by=c("userId","movieId")) %>%  
  mutate(pred= pred, residual = pred - rating) %>%
  arrange(desc(abs(residual))) %>% slice(1:100) %>% view

# update residual and pred columns with values from latest model
edx <- edx %>% left_join(predicted_train , by=c("userId","movieId")) %>%  
  mutate(pred= pred, residual = pred - rating) 

rm(predicted_train)
as_tibble(edx)

# limit scope
max_users <- 6000
max_movies <-1000
users_worst <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  arrange(desc(abs(residual))) %>% 
  select(userId)  %>%  unique %>% slice(1:max_users) %>% pull(userId)
movies_worst <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  arrange(desc(abs(residual))) %>% 
  select(movieId)  %>%  unique %>% slice(1:max_movies) %>% pull(movieId)


# put train test in the form of a matrix of residuals for N users x M movies
# replace the NA values with 0  //// ? \\\\\\

train_x <- edx %>% filter(movieId %in% movies_worst & userId %in% users_worst ) %>% 
  select(userId, movieId, residual) %>% 
  spread(movieId, residual, fill = 0)


rownames <- train_x$userId
train_x <- train_x[,-1] %>% as.matrix()
rownames(train_x) <- rownames
dim(train_x)

# SVD 
s <- svd(train_x)

s$d
s$u[1:10,1:10]
s$v[1:10,1:10]
dim(s$u)
dim(s$v)
dim(diag(s$d))

# You can check that the SVD works by typing:
# max (abs(s$u %*% diag(s$d) %*% t(s$v) - train_x))

# check the variance of components. 
# variance of Y = var (Y * V) = Var (U*D) = Var (D)
# 

# 90% of the variance
imax<-500
sum(s$d[1:imax]^2) / sum(s$d^2)

# ratio in number of parameters
imax*(nrow(s$v) +1 + nrow(s$u)) / ( nrow(train_x)*ncol(train_x) )

# reduced matrix
ur <- s$u[, 1:imax]
dr <- s$d[1:imax]
vr <- s$v[, 1:imax]

model_x <- sweep(ur, 2, dr, FUN="*") %*% t(vr)
# max(abs(train_x - model_x))

rownames(model_x) <- rownames(train_x)
colnames(model_x) <- colnames(train_x)

# train_x[1:10,1:10]
# model_x[1:10,1:10]

model_x <- as_tibble(model_x) %>% 
  mutate(userId=as.numeric(rownames(model_x))) %>% 
           gather(movieId, residualsvd, 1:ncol(model_x)) %>%
           mutate(movieId=as.numeric(movieId))
model_x %>% head

pred_train <- edx %>% 
  left_join(model_x , by = c("userId","movieId")) %>%  
  mutate (residualsvd =ifelse(is.na(residualsvd),0,residualsvd), 
          pred = pred - residualsvd) %>% 
  pull(pred)

# RMSE of predictions on the training set
RMSE(pred_train, edx$rating) 
# 0.7729606 for 6000x1000 sample

# assessment on validation set
#
predicted_val <- 
  pred_val %>%  
  left_join(model_x , by = c("userId","movieId")) %>% 
  mutate (residualsvd =ifelse(is.na(residualsvd),0,residualsvd), 
          pred = pred - residualsvd) %>% 
  pull(pred)

rmse_val <- RMSE(predicted_val, validation$rating) 
# 0.8493622 for 6000x1000

rmse_results <- bind_rows(rmse_results, data_frame(method="Additional effects from SVD factorisation ", RMSE = rmse_val ))
rmse_results %>% knitr::kable(digits = 6)


#   |method                                    |     RMSE|
#   |:-----------------------------------------|--------:|
#   |Movie & User effects with regularisation  | 0.864818|
#   |Additional User-Genre effect model        | 0.849530|
#   |Additional effects from SVD factorisation | 0.848963|

##
## conclusion: SVD brings a small amelioration but it seems not efficient to capture the correlations since we need to keep 
##
imax
# 1000 columns out of
length(s$d)
## 2115 to cover 90% of the variablility

## --------------------------------------------------------
## Other approaches that were explored with no success

##
## go back to user+movie + genre effects model
## try to apply a clustering approach

# we use again the matrix of residuals train_x, NA values filled with zero
train_x[1:10,1:10]

# we first explore a hierarchical clustering on users to check for a number of groups
d <- dist(train_x)
h <- hclust(d)
plot(h, cex = 0.65, main = "", xlab = "")

# try different values for the cutoff height => not realy able to separate efficiently because a lot of groups of 1 user vs. one large group
groups <- cutree(h, h = 30)
max(groups)
sapply(1:max(groups), function(x) {sum(groups==x)})  # count the elements in each group

# let's try on movies
d <- dist(t(train_x))
h <- hclust(d)
plot(h, cex = 0.65, main = "", xlab = "")

groups <- cutree(h, h = 50)
groups <- cutree(h, k = 5)
max(groups)
sapply(1:max(groups), function(x) {sum(groups==x)}) 
##
# same conclusion, clustering seems not succesfull at this stage
##

##
## explore the clustering of the training set users with kmeans
##
k <- kmeans(train_x, centers = 20, nstart = 5)
groups <- k$cluster
sapply(1:max(groups), function(x) {sum(groups==x)}) 

# check at the residual variability among each group
mean(edx$residual^2)
mean(matrixStats::colSds(train_x))

sgroups <- sapply(1:length(k$size), function (x) {as.numeric(names(groups)[groups==x] ) })
sapply(1:length(k$size), function (x) {
  edx %>% filter( userId %in% as.vector(sgroups[[x]]) ) %>% summarize(mean(residual^2), mean(abs(residual)))
})
ugroups <- data.frame(userId=as.numeric(names(groups)), group=as.vector(groups))

# check a sample of users with bad predictions and the associated group (reminder: residual were computed with user+movie+genre effects )
edx %>% filter(userId %in% users_worst & movieId %in% movies_worst) %>%
  filter(movieId %in% c(8,292,1089)) %>%
  left_join(ugroups, by="userId") %>% arrange(movieId, group) %>% head(50)

# 4 users in the same group have very different residual for a same movie
train_x[c("3504","6114","6562","10014"),"292"]
groups[c("3504","6114","6562","10014")]

edx %>% filter(userId %in% users_worst & movieId %in% movies_worst) %>%
  filter(movieId %in% c(8,292,1089)) %>%
  left_join(ugroups, by="userId") %>% 
  group_by(group,movieId) %>%
  summarize(n(), mean(residual), mean (abs(residual)), sd (residual)) %>% head(50) %>% view
# => the variance of residuals for a same movie in each group is quite high

edx %>% filter(userId %in% users_worst & movieId %in% movies_worst) %>%
  left_join(ugroups, by="userId") %>%
  group_by(group) %>% 
  summarize(n(), mean(residual), mean (abs(residual)), sd (residual)) %>% head(50) %>% view
  
# => the variance of residuals in each group is quite high

##
## explore the clustering of the training set movies (rather than users) with kmeans
##
k <- kmeans( t(train_x), centers = 20, nstart = 5)
groups <- k$cluster
k$size
mean(matrixStats::colSds(train_x))

sgroups <- sapply(1:length(k$size), function (x) {as.numeric(names(groups)[groups==x] ) })
sapply(1:length(k$size), function (x) {
  edx %>% filter( movieId %in% as.vector(sgroups[[x]]) & userId %in% users_worst) %>% summarize(sd(residual))
})
mgroups <- data.frame(movieId=as.numeric(names(groups)), group=as.vector(groups))

edx %>% filter(userId %in% users_worst & movieId %in% movies_worst) %>%
  filter(movieId %in% 200:500 & userId %in% 400:500) %>%
  left_join(mgroups, by="movieId") %>% 
  arrange(group, movieId, userId) %>% select(group, movieId, userId, residual)  %>% head(50)

edx %>% filter(userId %in% users_worst & movieId %in% sample(movies_worst,40)) %>%
  left_join(mgroups, by="movieId") %>% 
#  group_by(group,movieId) %>%
#  summarize(n(), mean(residual), mean (abs(residual)), sd (residual)) %>% 
  arrange(group)%>%
  ggplot(aes(group=movieId, residual, fill = group)) +
  geom_boxplot()


# 
# New try: random forest regression on residuals of the movie-user-genre effects
# 
# 
library(randomForest)

# 
# Limit the training to a subset of worst ratings + round the residual
edx_worst <- edx %>% filter(movieId %in% movies_worst & userId %in% users_worst ) %>%
  left_join(ugroups, by="userId") %>% 
  left_join(mgroups, by="movieId") %>% 
  mutate(movieId=factor(movieId),userId=factor(userId), ugroup=factor(group.x),mgroup=factor(group.y),
         roundres = round(2*(pred-rating),0)/2 )%>%
  select(userId,movieId,ugroup,mgroup,residual, roundres)

str(edx_worst)

# RF on the user and movies is not possible because the RF does not support more than 53 categories
# RF on the b_u and b_i is not possible because requires too much memory
# therefore test RF on the user groups and movie groups computed before with kmeans

fit <- randomForest(residual~mgroup+ugroup, data = edx_worst, ntree=5, mtry=2, maxnodes=30) 
plot(fit)

y_hat <- predict(fit, newdata = edx_worst)
RMSE(y_hat, edx_worst$residual) # 0.93

# reg tree rpart on groups

train_rpart <- train(residual~mgroup+ugroup, data = edx_worst, 
                     method = "rpart")
plot(train_rpart)


## cluster users and movies in groups 
##
k <- kmeans(train_x, centers = 60, nstart = 5)
groups <- k$cluster
ugroups <- data.frame(userId=as.numeric(names(groups)), group=as.vector(groups))

km <- kmeans( t(train_x), centers = 10, nstart = 5)
mgroups <- data.frame(movieId=as.numeric(names(k$cluster)), group=as.vector(k$cluster))

str(k$centers)

train_x[1:5,1:5]
head(ugroups)
ugroups[1:5,2]
k$centers[ugroups[1:5,2],1:5]
rowMeans(train_x)[1:5]
rowMeans(k$centers[ugroups[1:5,2],])


#
# random selection matrix (10% of users)
#
users_rand <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  filter( userId %in% sample(unique(userId), 6000) ) %>% 
  pull(userId) %>% unique
length(users_rand)

rand_x <- edx %>% filter(userId %in% users_rand) %>% 
  select(userId, movieId, residual) %>% 
  spread(movieId, residual, fill = 0)

#
# matrix reduction 

rownames <- rand_x$userId
rand_x <- rand_x[,-1] %>% as.matrix()
rownames(rand_x) <- rownames
dim(rand_x)

s<-svd(rand_x)

# 90% of the variance
imax<-1000
sum(s$d[1:imax]^2) / sum(s$d^2)

# ratio in number of parameters
imax*(nrow(s$v) +1 + nrow(s$u)) / ( nrow(rand_x)*ncol(rand_x) )

# reduced matrix
ur <- s$u[, 1:imax]
dr <- s$d[1:imax]
vr <- s$v[, 1:imax]

model_x <- sweep(ur, 2, dr, FUN="*") %*% t(vr)
# max(abs(train_x - model_x))

rownames(model_x) <- rownames(rand_x)
colnames(model_x) <- colnames(rand_x)

model_x <- as_tibble(model_x) %>% 
  mutate(userId=as.numeric(rownames(model_x))) %>% 
  gather(movieId, residualsvd, 1:ncol(model_x)) %>%
  mutate(movieId=as.numeric(movieId))

# RMSE of predictions on the training set
pred_train <- edx %>% 
  left_join(model_x , by = c("userId","movieId")) %>%  
  mutate (residualsvd =ifelse(is.na(residualsvd),0,residualsvd), 
          pred = pred - residualsvd) %>% 
  pull(pred)

RMSE(pred_train, edx$rating) # 0.7762015

# RMSE on validation set
predicted_val <- 
  pred_val %>%  
  left_join(model_x , by = c("userId","movieId")) %>% 
  mutate (residualsvd =ifelse(is.na(residualsvd),0,residualsvd), 
          pred = pred - residualsvd) %>% 
  pull(pred)

rmse_val <- RMSE(predicted_val, validation$rating) 
# 0.84947 for 6000x9423, poorer than SVD on worst subset.


