from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder.appName("Project22_Diabetes_Engine").getOrCreate()

# 1. Load the diabetes dataset
df = spark.read.csv("diabetes.csv", header=True, inferSchema=True)
print("\n--- CHECKPOINT 1: RAW DATA LOADED ---")
df.show(5)

# 2. Handle missing values (Logical Zeros)
cols_to_fix = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]
for col_name in cols_to_fix:
    mean_val = df.select(F.mean(col_name)).collect()[0][0]
    df = df.withColumn(col_name, F.when(F.col(col_name) == 0, mean_val).otherwise(F.col(col_name)))

print("\n--- CHECKPOINT 2: DATA CLEANED (Mean Imputation Applied) ---")

# 3. Standardize numerical features
feature_cols = [c for c in df.columns if c != 'Outcome']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df_vect = assembler.transform(df)

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_vect)
df_final = scaler_model.transform(df_vect)
print("\n--- CHECKPOINT 3: FEATURES VECTORIZED & STANDARDIZED ---")

# 4. Build Random Forest classifier
train, test = df_final.randomSplit([0.8, 0.2], seed=42)
rf = RandomForestClassifier(labelCol="Outcome", featuresCol="features", numTrees=100)
model = rf.fit(train)

# 5. Evaluate model performance
preds = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="Outcome")

metrics = {
    "F1-Score": evaluator.setMetricName("f1").evaluate(preds),
    "Precision": evaluator.setMetricName("weightedPrecision").evaluate(preds),
    "Recall": evaluator.setMetricName("weightedRecall").evaluate(preds)
}

# 6. Save results for the Dashboard
pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)
importances = pd.DataFrame({'Feature': feature_cols, 'Score': model.featureImportances.toArray()})
importances.to_csv("importance.csv", index=False)

print("\n" + "="*40)
print("✅ PIPELINE EXECUTION COMPLETE")
print(f"Final F1-Score: {metrics['F1-Score']:.4f}")
print("="*40 + "\n")

spark.stop()