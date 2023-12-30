### PART I - PREDICTING AVERAGE PRICE (ALINA) ----------------------
# Libraries used:
library(skimr)
library(ggplot2)
library(rsample)
library(recipes)
library(caret)
library(visdat)
library(DataExplorer)
library(recipes)
library(vip)
library(boot)
library(lubridate)


#a) Upload the dataset, treat character columns as factors 
pumpkin <- read.csv("/Users/aleksandrakusz/Desktop/1st BI/ML1/Datasets/US-pumpkins.csv", 
                    stringsAsFactors = TRUE, na.strings = c("", " ", "NA"))

#b) Get a glimpse of the data, check variable types etc
skim(pumpkin) #check basic properties
dim(pumpkin) #getting the dimension of the data 
str(pumpkin) #checking variable types 
head(pumpkin) #first 6 rows 
class(pumpkin) #check variable type
#To deal with imperfections and noise the procedures would be: 
# -Handling missing values (e.g., using functions like is.na() or packages like mice)
# -Detecting and treating outliers (e.g., using boxplot() or packages like outliers)
# -Data normalization or standardization (e.g., using functions like scale())
# -Encoding categorical variables (e.g., using factor() or packages like dummies)
# -Feature scaling (e.g., using functions like scale())

#c) Delete the unknown variables: X and X.1 
pumpkin <- subset(pumpkin, select = -c(X, X.1))
str(pumpkin)

#d) Extract the month and the year into the new columns from the Date variable and remove the orginal varibale. 
#Change the format of a date variable from factor to date 

pumpkin <- pumpkin %>%
  mutate(Date = mdy(Date), month = month(Date)) %>% dplyr::select(-Date)

#e) Evaluate the missing data. 
#Consider removing variables with an extreme number of missing values from the dataset.
sum(is.na(pumpkin))

# Visually
vis_miss(pumpkin, cluster = TRUE)
#Variables that have the most missing values: Grade, Environment, Quality, Condition, Appearance, Storage, Crop, Trans.Mode

plot_missing(pumpkin)

#Removing variables that have more than 80% missing values 
pumpkin <- subset(pumpkin, select = -c(Trans.Mode, Crop, Storage, Appearance, Condition, Quality, Environment, Grade, Sub.Variety, Unit.of.Sale,Origin.District,Type))
str(pumpkin)

#The rest of the variables that have less than 80% of missing values will be trated with the imputation method like knn that finds the most similar value, but has high computation. 
#K-nearest neighbor (KNN) imputes values by identifying observations with missing values, 
#then identifying other observations that are most similar based on the other available features, 
#and using the values from these nearest neighbor observations to impute missing values.

# f) Create the dependent varibale - Average Price = (Low Price + High Price )/2
# Evaluate the distribution and check transformations to normalize it and decide on the most appropriate transformation. 

pumpkin$AvgPrice <- (pumpkin$Low.Price+pumpkin$High.Price)/2
pumpkin <- subset(pumpkin, select = -c(Low.Price,High.Price))

par(mfrow=c(1,3))
hist(pumpkin$AvgPrice, breaks = 20, col = "red", border = "red") 

# log-transformation (if all values are positive)
transformed_response <- log(pumpkin$AvgPrice)

# Box-Cox transformation (lambda=0 is equivalet to log(x))
transformed_response_BC <- forecast::BoxCox(pumpkin$AvgPrice, lambda="auto")

#Yeo 
library(car)
transformed_response_yeo <- yjPower(pumpkin$AvgPrice, lambda = 0.5)

hist(transformed_response_yeo, breaks = 20, col = "pink", border = "pink")
hist(transformed_response, breaks = 20, col = "lightgreen", border = "lightgreen") #Transformed with log 
hist(transformed_response_BC, breaks = 20, col = "lightblue", border = "lightblue") #Transformed with BC

#We have the outlier regardless the transformation, the most appropriate would be the Yeo - Johson 

#g) Evaluate the features with zero and near - zero variance and decide which variables will be eliminated. 
caret::nearZeroVar(pumpkin, saveMetrics = TRUE) %>% 
  tibble::rownames_to_column() %>% 
  filter(nzv)

#The variable "Repack" is identified as having near - zero variance.
#This means that most of its values are likely the same or very similar, and it might not provide much information for modeling or analysis purposes.

#h) Display the distributions of the numeric features. What type of pre-processing to be implemented later.
plot_histogram(pumpkin)
# Based on this output , we can make individual data processing for each variable
# Alternatively, we will: 
# Normalize all numeric columns using the blueprint
#  e.g., data_recipe %>% step_YeoJohnson(all_numeric())  
# Standardize all numeric values using the blueprint
# e.g., data_recipe %>%
# step_center(all_numeric(), -all_outcomes()) %>%
# step_scale(all_numeric(), -all_outcomes())

# Scales are similiar - no need to standardise
# Since AvgPrice will be transformed to YeoJohson, we can do the same to all
# the numeric columns
# There are some outliers, should be watched for later impact
# NAs will be imputed as mentioned previously


#i) Display the distribution of factor features. Decide which type of pre-processing should be implemented. 
plot_bar(pumpkin)

# a few categories can be lumped in a few columns
# Repack is 99% one variable, it was previously mentioned that it has the near zero variance, so it will be removed. 

#j) Proceed by creating the blueprint that will prepare the data to predict the average price for pumpkin. 
#Set up the recipe including all the steps desired for data pre-processing. Prepare and bake both the training and test data. 
#The end result should be two datasets: baked_train and baked_test, which will be ready for the further analysis. 
#Display the dimensions of these new datasets. 

#Splitting the data set into training and test. 
set.seed(123)
split <- initial_split(pumpkin, prop = 0.8, 
                       strata = "AvgPrice")
pumpkin_train  <- training(split)
pumpkin_test   <- testing(split)

#Creatng blueprint 
pumpkin_recipe <- recipe(AvgPrice ~ ., data = pumpkin_train) %>%
  step_impute_knn(all_predictors(), neighbors = 6)%>% #everything that I am measuring but the target 
  step_center(all_numeric(), -all_outcomes()) %>% #all numeric that are not the y (avgprice)
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_YeoJohnson(all_numeric()) %>%
  step_nzv(all_nominal(), all_numeric()) %>%
  step_other(Origin, threshold = 0.01, other ="Other") %>%
  step_other(Package, threshold = 0.01, other = "Other") %>%
  step_dummy(all_nominal(), one_hot = TRUE)

pumpkin_recipe
# prepare
prepare <- prep(pumpkin_recipe, training = pumpkin_train)
prepare

# bake
baked_train <- bake(prepare, new_data = pumpkin_train)
baked_test <- bake(prepare, new_data = pumpkin_test)
baked_train
baked_test  

sum(is.na(baked_train))

#k) Using the caret library, with k-fold cross-validation, and the data prepared 
#before using the blueprint, train a set of regression models to predict 
#the average price of pumpkin in the US market.

trControl = trainControl(method = "cv", number = 10)

LM_model <- train(AvgPrice ~ ., 
                  data = baked_train, 
                  method = "lm", 
                  trControl = trControl)

KNN_model <- train(AvgPrice ~ ., 
                   data = baked_train, 
                   method = "knn", 
                   trControl = trControl)

GLM_model <- train(AvgPrice ~ ., 
                   data = baked_train, 
                   method = "glm", 
                   trControl = trControl)

#l) Summarize the results for all models and select the best one. 
results <- resamples(list(LM_model, KNN_model, GLM_model))
summary(results)
#We should look at the model with the lowest MAE, lowest RMSE, and highest RSquared. 
#Model2 appears to be the best among the three, as it has the lowest MAE and RMSE values and the highest R-squared value, 
#indicating better accuracy and fit to the data compared to the other models.

#m) Evaluate and interpret the error in the test data for the best model.
pred = predict(KNN_model, baked_test)
postResample(pred, baked_test$AvgPrice)

#n) Assess the feature importance for the top performing model and 
#draw conclusions regarding the most influential predictors of the price. 
vip::vip(LM_model, num_features = 20, method = "model")
#Using the second best model as KNN is not compatible with the vip package

#o) With the objective of reducing errors, adjust the previous code and subsequently re-run the models.
#Perform a comparative analysis of the results to assess whether these modifications lead to a
#reduction in errors.

###PART II - BEN ---------------------------------------------------------------
#a) Load the data file into R. 
#Keep only variables City Name, Variety, Date, Low Price, High Price and Item Size. 
#Delete other variables.
pumpkin_2 <- read.csv("/Users/aleksandrakusz/Desktop/1st BI/ML1/Datasets/US-pumpkins.csv", 
                      stringsAsFactors = TRUE,
                   na.strings = c("", " ", "NA"))
pumpkin_2 <- pumpkin_2[, c("City.Name", "Variety", "Date", "Low.Price", "High.Price", "Item.Size")]
head(pumpkin_2)

#b) Add two new variables: Price=(Low Price+High Price)/2 and Spread=5+ (High Price-Low Price). 
#Then, delete the Low Price and High Price variables. 
#Provide histograms and density plots for Price and Spread variables using ggplot2 library. 
#Discuss similarities and differences in these plots.

pumpkin_2$Price <- (pumpkin_2$Low.Price+pumpkin_2$High.Price)/2
pumpkin_2$Spread <- 5+(pumpkin_2$High.Price - pumpkin_2$Low.Price)
pumpkin_2 <- subset(pumpkin_2, select = -c(Low.Price,High.Price))

# Histogram and Density Plot for Price
ggplot(pumpkin_2, aes(x = Price)) +
  geom_histogram(binwidth = 10, fill = "blue", color = "white", alpha = 0.7) 
  labs(title = "Distribution of Price", x = "Price", y = "Density")

# Histogram and Density Plot for Spread
ggplot(pumpkin_2, aes(x = Spread)) +
  geom_histogram(binwidth = 1, fill = "green", color = "white", alpha = 0.7)
  labs(title = "Distribution of Spread", x = "Spread", y = "Density")
  
library(scales)
ggplot(pumpkin_2) +
  geom_density(aes(x=Price))
#c) Add two more new variables: the month and the year in which the product was sold, 
#and then delete the Date variable. 

pumpkin_2 <- pumpkin_2 %>%
  mutate(Date = mdy(Date), month = month(Date), year = year(Date)) %>% dplyr::select(-Date)

#d) Remove rows that have “ “ (i.e. blank) in either variables Variety or Item Size.
# Identify row numbers with NA values in either Variety or Item Size
missing_row_numbers <- which(is.na(pumpkin_2$Variety) | is.na(pumpkin_2$Item.Size))

# Print the row numbers with missing values
cat("Row numbers with missing values:", missing_row_numbers, "\n")
sum(is.na(pumpkin_2)) #284 rows 
# Remove rows with identified row numbers
pumpkin_2 <- pumpkin_2[-missing_row_numbers, ] #should be 1473

#e) Suppose that you are a data scientist in a company and the sales department asks more info 
#from you on AdjPrice=median(Price_i*Spread_i)/mean(Spread_i), where median and mean are taken 
#over all units (i) in the sample. On this original dataset and without resampling, 
#could you provide a point estimate and a standard error for this AdjPrice measure? 
pumpkin_2$AdjPrice=median(pumpkin_2$Price*pumpkin_2$Spread)/mean(pumpkin_2$Spread)


#f) Set your seed. Use B=1000 bootstrap samples to provide a point estimate, 
#a standard error, and a 95% confidence interval for this measure. Briefly discuss your results.
set.seed(123)

result <- function(data, index) {
  median(pumpkin_2$Price[index] * pumpkin_2$Spread[index]) / mean(pumpkin_2$Spread[index])
}

library(boot)
boot_result <- boot(data = pumpkin_2, statistic = result, R = 1000)
boot.ci(boot_result, type = "basic")

#g) Split data into training set and test set according to year<2017 and year=2017, respectively.

train_data <- subset(pumpkin_2, year < 2017)
test_data <- subset(pumpkin_2, year == 2017)

#h) In the prediction task, the response variable is Price, while other variables 
#are predictors. In a linear model and on a training set, estimate (i) OLS, 
#((ii) ridge with 5-fold cross validation, 
#(iii) lasso with leave one out cross validation.

#i)
model1 <- lm(Price ~., data = train_data)
plot(model1) 
summary(model1)

#ii) 
install.packages("glmnet")
library(glmnet)
X <- model.matrix(~ City.Name + Variety + Item.Size + month + year + Spread - 1, data = train_data)

str(train_data)
cv.out_k <- cv.glmnet(X, train_data$Price, alpha = 0, nfolds = 5)
bestlam_k<-cv.out_k$lambda.min

#iii) 
n <- length(train_data)  #sample size
cv.out_l<-cv.glmnet(X,train_data$Price,alpha=1, nfolds = n)
bestlam_L <- cv.out_l$lambda.min

#j)
lm_intercept <- lm(Price ~ 1, data = train_data)
summary(lm_intercept)
lm_city <- lm(Price ~ City.Name, data = train_data)

lm_month <- lm(Price ~ month, data = train_data)

lm_month_city <- lm(Price ~ month + City.Name, data = train_data)


# i. In addition, estimate the following benchmark linear models on the training set: (i) no predictor
# model (intercept only), (ii) City Name, (iii) month, (iv) City Name and month. These are regressions
# with at most two predictors.

predictions_1 <- predict(lm_intercept, newdata = test_data)
predictions_2 <- predict(lm_city, newdata = test_data)
predictions_3 <- predict(lm_month, newdata = test_data)
predictions_4 <- predict(lm_month_city, newdata = test_data)


predictions_K <- predict(cv.out_k, newx = X, s = bestlam_k)
predictions_L <- predict(cv.out_l, newx = X, s = bestlam_L)

actual_values <- test_data$Price

mse_1 <- mean((actual_values - predictions_1)^2)
mse_2 <- mean((actual_values - predictions_2)^2)
mse_3 <- mean((actual_values - predictions_3)^2)
mse_4 <- mean((actual_values - predictions_4)^2)


mse_K <- mean((train_data$Price - predictions_K)^2)
mse_L <- mean((train_data$Price - predictions_L)^2)

# Calculate MSE for each set of predictions
mse_ridge <- mean((test_set$Price - predict_ridge)^2)
mse_lasso <- mean((test_set$Price - predict_lasso)^2)
mse_intercept <- mean((test_set$Price - predict_intercept)^2)
mse_city <- mean((test_set$Price - predict_city)^2)
mse_month <- mean((test_set$Price - predict_month)^2)
mse_city_month <- mean((test_set$Price - predict_city_month)^2)

# Combine all MSEs into a named vector for easy comparison
mse_values <- c(
                Ridge = mse_K,
                Lasso = mse_L,
                Intercept = mse_1,
                City = mse_2,
                Month = mse_3,
                City_Month = mse_4)

# Print the MSE values
print(mse_values)

# Ordering the MSE values from smallest to largest
ordered_mse <- sort(mse_values)

# Print the ordered MSE values
ordered_mse


