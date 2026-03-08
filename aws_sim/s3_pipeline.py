import boto3
import pandas as pd
import os
import sys

# Connect to LocalStack S3
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    aws_access_key_id='test',
    aws_secret_access_key='test',
    region_name='us-east-1'
)

BUCKET_NAME = 'spotify-data-lake'

def create_bucket():
    s3.create_bucket(Bucket=BUCKET_NAME)
    print(f"✅ Created S3 bucket: {BUCKET_NAME}")

def upload_data():
    files = {
        'raw/dataset.csv': 'data/raw/dataset.csv',
        'processed/processed.csv': 'data/processed.csv',
        'clustered/clustered.csv': 'data/clustered.csv'
    }
    for s3_key, local_path in files.items():
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"✅ Uploaded {local_path} → s3://{BUCKET_NAME}/{s3_key}")

def list_bucket():
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    print(f"\n📦 Files in s3://{BUCKET_NAME}:")
    for obj in response.get('Contents', []):
        size_mb = obj['Size'] / (1024 * 1024)
        print(f"   {obj['Key']} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    create_bucket()
    upload_data()
    list_bucket()
    print("\n🎉 AWS S3 simulation complete!")