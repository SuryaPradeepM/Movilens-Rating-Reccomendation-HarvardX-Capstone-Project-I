########################################################################################### 
# PH125.9x Capstone-Part(I) MovieLens Rating Prediction Project                           #
#                                                                                         #
# HarvardX DataScience Course,                                                            #
#                                                                                         #
# Capstone Project part 1: MovieLens Prediction.                                          #
#                                                                                         #
# R Script to build models and create predictions for the movie ratings test set.         #
###########################################################################################


# MovieLens Rating Prediction Project Code 

# Install and load necessary dependencies
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")


#### Introduction ####

## Dataset ##

# Download and generate edx and validation datasets

################################
# Create edx set, validation set
################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Note: this process could take a couple of minutes
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

# Delete the unnecessary dataframes created to free up memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)


### Exploratory Data Analysis ###

# Look at a few starting rows
head(edx) %>% print.data.frame()

# structure
str(edx)

# Summary 
summary(edx)

# No.of unique movies and users in the edx dataset 
edx %>% summarize(n_users = n_distinct(userId), 
                  n_movies = n_distinct(movieId))


# Highest Rated movies
edx %>% group_by(title) %>%
  summarize(numberOfRatings = n(), averageRating = mean(rating)) %>%
  arrange(desc(averageRating)) %>%
  top_n(10, wt=averageRating)

# Highest Rated movies with atleast 100 ratings
edx %>% group_by(title) %>%
  summarize(numberOfRatings = n(), averageRating = mean(rating)) %>%
  filter(numberOfRatings > 100) %>%
  arrange(desc(averageRating)) %>%
  top_n(10, wt=averageRating)

# Extract unique genres with seperate_rows and arrange them in descending order
genres <- edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(n = n()) %>%
  arrange(desc(n))

genres %>% print.data.frame()


# Generate histogram of ratings from the dataset
edx %>% ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "#51A8C9") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Distribution of Ratings") + 
  xlab("Count of ratings") +
  scale_colour_wsj("colors6", "") +
  theme_wsj(base_size = 10, color = "blue", 
            base_family = "sans", title_family = "sans")


# Generate log scaled frequency distribution of movies and ratings
edx %>% count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color = "#51A8C9") +
  scale_x_log10() +
  ggtitle("Frequency distribution of Ratings for Movies") + 
  scale_colour_wsj("colors6", "") +
  theme_wsj(base_size = 10, color = "blue", 
            base_family = "sans", title_family = "sans")


# Find the movies with only a single user rating (Outliers)
edx %>% group_by(movieId) %>%
  summarize(ratings = n()) %>%
  filter(ratings == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = ratings) %>%
  slice(1:20) %>%
  knitr::kable()

# Generate the log scaled frequency distribution of users and ratings
edx %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color = "#51A8C9") +
  scale_x_log10() +
  ggtitle("Frequency distribution of Ratings for Users") +
  scale_colour_wsj("colors6", "") +
  theme_wsj(base_size = 10, color = "blue", 
            base_family = "sans", title_family = "sans")


# Generate mean ratings of users who have rated atleast 100 movies
edx %>% group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 40, color = "#51A8C9") +
  xlab("Average rating") +
  ylab("Count of users") +
  ggtitle("Average Ratings by Number of Users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_colour_wsj("colors6", "") +
  theme_wsj(base_size = 10, color = "blue", 
            base_family = "sans", title_family = "sans")


# Modelling Approach #

# Root of mean of squared errors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Avg. movie rating model ##

# Compute the dataset's mean rating
mu <- mean(edx$rating)
mu


# Rmse for average rating model
naive_rmse <- RMSE(validation$rating, mu)
naive_rmse

# Check results
# Save prediction in data frame
rmse_results <- tibble(Model = "Naïve Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

## Movie effect model ##

# Simple model taking into account the movie effect b_i
# Subtract the rating minus the mean for each rating for a movie
# Plot no.of movies with the computed b_i
# Compute the bias terms of movie ratings
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# plot distribution
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("#51A8C9"),
                     main = "Number of movies for respective computed bias") + 
                      scale_colour_wsj("colors6", "") +
                      theme_wsj(base_size = 10, color = "blue", 
                        base_family = "sans", title_family = "sans")

## Test and save rmse results 
# Left join moive_avgs on key movieId
predictions <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect_rmse <- RMSE(predictions, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Model="Movie effect model",  
                                 RMSE = movie_effect_rmse ))

rmse_results %>% knitr::kable()

## Movie and user effect model ##

# Plot penaly term user effect #
# Compute the bias terms of user ratings
user_avgs<- edx %>% left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))

# plot distribution
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("#51A8C9"),
                   main = "Number of users for respective computed bias")  + 
             scale_colour_wsj("colors6", "") +
             theme_wsj(base_size = 10, color = "blue", 
                        base_family = "sans", title_family = "sans")


# Compute user averages
user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

## Test and save the rmse results 
# predictions for this model
predictions <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_effect_rmse <- RMSE(predictions, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Model="Movie & User effect model",  
                                     RMSE = user_effect_rmse))
rmse_results %>% knitr::kable()

## Regularised movie and user effect model ##

# lambda is a tuning parameter
# Trying different values for the regularization term
lambdas <- seq(2, 10, 0.25)
rmses <- sapply(lambdas, function(lambda){
  movie_avg <- mean(edx$rating)
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})


# Plot rmses vs lambdas to select the optimal omega                                                             
qplot(lambdas, rmses,  color = I("#51A8FF"), 
      main = "RMSEs vs. Lambdas") + 
      theme_fivethirtyeight()


# get the lamba for minimum value of rmse
lambda <- lambdas[which.min(rmses)]
lambda

# Test and save results                                                             
rmse_results <- bind_rows(rmse_results,
                          tibble(Model="Regularisation & Movie & User effect model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

#### Results ####                                                            
# RMSE results overview                                                          
rmse_results %>% knitr::kable()
