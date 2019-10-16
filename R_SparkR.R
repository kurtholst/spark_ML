# Developing Machine Learning models with RStudio on Azure Databricks using Spark in memory training.

# Machine learning with Spark – all Interaction performed directly from R API SparkR or SparklyR
# 
# The use case presented in this blog is a rather simple dataset. 
# I have chosen this dataset so that you potentially can re-do the steps I am showing in your own environment. 
# The data science challenge is a classic supervised binary classification. 
# Given a number of features all with certain characteristics, our goal is to build a machine learning model to 
# identify people affected by type 2 diabetes.
# 
# Data description:
# pregnant: Number of times pregnant
# glucose: GlucosePlasma glucose concentration a 2 hours in an oral glucose tolerance test
# pressure: BloodPressureDiastolic blood pressure (mm Hg)
# triceps: SkinThicknessTriceps skin fold thickness (mm)
# insulin: Insulin2-Hour serum (mu U/ml)
# mass: BMIBody mass index (weight in kg/(height in m)^2)
# pedigree: Diabetes pedigree function
# age: Age in years
# diabetes: Outcome Class variable (0 or 1). 268 of 768 are 1, the remaining are 0.


rm(list=ls())

# Install packages if needed.
if (!require('mlbench')) install.packages('mlbench'); 
if (!require('caTools')) install.packages('caTools'); 
if (!require('SparkR')) install.packages('SparkR'); 

# Load libraries
library(mlbench)
library(SparkR)
library(caTools)

# Spark connection
#SparkConnection <- sparkR.session() # initiate Spark contex
SparkConnection <- sparkR.init()
str(SparkConnection)
sparkR.session()

# Load data
data("PimaIndiansDiabetes")
diabetes <- PimaIndiansDiabetes

# Save data to Spark
Spark_df <- createDataFrame(x = diabetes, schema = NULL)  # In Spark 1.6 and earlier, use `sqlContext` instead of `spark`
str(Spark_df)

# Split data
df_list <- randomSplit(x = Spark_df, weights = c(0.8, 0.2), seed = 12345)
str(df_list[1])
train <- df_list[1]
# TO-DO proper split

features <- as.data.frame(c("pregnant", "glucose", "pressure", "triceps", "insulin","mass", "pedigree", "age"))
Spark_features <- createDataFrame(x = features, schema = NULL)  # In Spark 1.6 and earlier, use `sqlContext` instead of `spark`

# Train model.
# Fit a random forest classification model with spark.randomForest
model <- spark.randomForest(data = Spark_df, 
                            formula = diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass + pedigree + age ,  
                            type = "classification", 
                            numTrees = 10)

# Model summary
summary(model)

# Prediction
predictions <- predict(model, Spark_df)
head(predictions)

predictions <- predict(model, Spark_df)


# save the model
getwd()
#path <- "path/to/model"
path <- "/home/kurho@dmpmst.onmicrosoft.com/R/spark_ML"
#write.ml(model, path)
write.ml(object = model, path = path)

# Load model
savedModel <- read.ml(path)
summary(savedModel)

# Apply the saved model to 
predictions_new <- predict(object = savedModel, Spark_df)

# Fetch data to local R session
predictions_local <- collect(predictions_new)

head(predictions_local)
