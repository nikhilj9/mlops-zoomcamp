#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from prefect import task, flow

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)
    
    print(f"Number of records loaded: {len(df)}")
    print(f"Dataset shape: {df.shape}")
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    print(f"After data preparation - Number of records: {len(df)}")
    print(f"After data preparation - Dataset shape: {df.shape}")
    
    return df


@task(timeout_seconds=300, retries=1) 
def train_model(df_processed):
    
    categorical = ['PULocationID', 'DOLocationID']
    numerical = []
    
    dicts = df_processed[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)
    
    target = 'duration'
    y_train = df_processed[target].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    print(f"Model intercept: {lr.intercept_:.2f}")
    
    return dv, lr


@task
def log_model_to_mlflow(dv, lr):

    mlflow.set_tracking_uri("file:./mlruns")
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model",
            registered_model_name="taxi-duration-model"
        )
        
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID, DOLocationID")
        mlflow.log_metric("intercept", lr.intercept_)
        
        run_info = mlflow.active_run().info
        run_id = run_info.run_id

    return run_id


@flow
def taxi_duration_training_flow(year: int = 2023, month: int = 3):

    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    
    df_processed = read_dataframe(filename)
    df_processed = df_processed.sample(n=100000, random_state=42)
    dv, lr = train_model(df_processed)
    run_id = log_model_to_mlflow(dv, lr)
    
    return run_id


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=3)
    args = parser.parse_args()
    
    run_id = taxi_duration_training_flow(year=args.year, month=args.month)
    print(f"MLflow run_id: {run_id}")
