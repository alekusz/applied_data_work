### PART I - DATA UNDERSTANDING, ENGINEERING AND ANALYSIS ----------------------
# Libraries used:
library(skimr)
library(ggplot2)
library(rsample)
library(recipes)
library(caret)
library(visdat)
library(DataExplorer)
library(recipes)

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
                  data = baked_test, 
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




