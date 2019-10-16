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

# Load and Install packages if needed.
if (!require('mlbench')) install.packages('mlbench'); 
if (!require('caTools')) install.packages('caTools'); 
if (!require('SparkR')) install.packages('SparkR'); 

# Spark connection
#SparkConnection <- sparkR.session() # initiate Spark contex
SparkConnection <- sparkR.init()
str(SparkConnection)
sparkR.session()

# Load data
data("PimaIndiansDiabetes")
diabetes <- PimaIndiansDiabetes

# Save data to Spark
Path <- "/home/kurho@dmpmst.onmicrosoft.com/R/spark_ML/data"

Spark_train <- createDataFrame(x = diabetes[0:700,], schema = NULL)  # In Spark 1.6 and earlier, use `sqlContext` instead of `spark`
Spark_test  <- createDataFrame(x = diabetes[701:768,], schema = NULL)  
write.df(Spark_train, path = Path, source = NULL, mode = "overwrite")
write.df(Spark_test, path = Path, source = NULL, mode = "overwrite")

# Explore data
printSchema(Spark_train)

# Create a df consisting of only the 'age' column using a Spark SQL query
age <- sql("SELECT age FROM Spark_train")

str(Spark_train)
str(Spark_test)

################
# Train model. #
# Fit a random forest classification model with spark.randomForest
model <- spark.randomForest(data = Spark_train, 
                            formula = diabetes ~ pregnant + glucose + pressure + 
                              triceps + insulin + mass + pedigree + age ,  
                            type = "classification", 
                            numTrees = 500)
# Model summary
summary(model)

# Prediction on test data
predictions <- predict(model, Spark_test)
head(predictions)

# save the model
#path <- "path/to/model"
Path <- "/home/kurho@dmpmst.onmicrosoft.com/R/spark_ML/models"
#write.ml(model, path)
write.ml(object = model, path = Path)
write.overwrite().save(
  rf_model.write().overwrite().save(rf_model_path)
  
  
# Load model
savedModel <- read.ml(path = Path)
summary(savedModel)

# Apply the saved model to 
predictions_new <- predict(object = savedModel, Spark_df)

# Fetch data to local R session
predictions_local <- collect(predictions_new)

head(predictions_local)



######################## 