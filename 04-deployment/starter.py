#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys
from prefect import flow, task

@task
def get_input_args():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    return year, month

@task
def load_model(model_path='model.bin'):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

@task
def read_data(year, month):
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = pd.read_parquet(filename)
    return df

@task
def preprocess_data(df, categorical_columns):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical_columns] = df[categorical_columns].fillna(-1).astype('int').astype('str')
    return df

@task
def transform_features(df, dv, categorical_columns):
    dicts = df[categorical_columns].to_dict(orient='records')
    X_val = dv.transform(dicts)
    return X_val

@task
def make_predictions(X_val, model):
    y_pred = model.predict(X_val)
    return y_pred

@task
def aggregate_results(y_pred):
    std_dev = y_pred.std()
    mean_val = y_pred.mean()
    print(f"Std: {std_dev}")
    print(f"Mean: {mean_val}")
    return std_dev, mean_val

@task
def save_results(df, y_pred, year, month, output_file='result.parquet'):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'duration': y_pred})
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return df_result

@flow(name="batch-inference")
def batch_inference_flow():

    year, month = get_input_args()
    
    dv, model = load_model()
    
    df = read_data(year, month)
    categorical_columns=['PULocationID', 'DOLocationID']
    df_processed = preprocess_data(df, categorical_columns)
    
    X_val = transform_features(df_processed, dv, categorical_columns)
    y_pred = make_predictions(X_val, model)
    
    aggregate_results(y_pred)
    save_results(df_processed, y_pred, year, month)

if __name__ == '__main__':
    batch_inference_flow()