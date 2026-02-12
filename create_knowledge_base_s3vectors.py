import boto3
import json
import logging
import os
import re
import requests
import time
import uuid
import zipfile


from knowledge_base_s3 import BedrockKnowledgeBase
from tqdm import tqdm
from urllib.parse import urlparse


logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

session = boto3.session.Session()
region = os.getenv('AWS_REGION', session.region_name)
if region:
    print(f'Region: {region}')
else:
    print("Cannot determine AWS region from `AWS_REGION` environment variable or from `boto3.session.Session().region_name`")
    raise

s3_client = boto3.client('s3', region)
bedrock_agent_client = boto3.client('bedrock-agent', region)
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region)
logs_client = boto3.client('logs', region)


def download_file(url):
    destination = os.path.basename(urlparse(url).path)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        file_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_zip_file(zip_path):
    try:
        n_files = 0
        with zipfile.ZipFile(zip_path, 'r') as f:
            file_list = f.namelist()
            print(f"Extracting files from {zip_path}")
            for file in tqdm(file_list, desc="Extracting"):
                if file.startswith('__') or '.DS_Store' in file:
                    print(f'Skipping file: {file}')
                    continue
                clean_filename = re.sub(r'[\s-]+', '-', file).lower()
                f.extract(file, '.')
                os.rename(file, clean_filename)
                n_files += 1
        print(f"Successfully extracted {n_files} files")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def create_s3_bucket_with_random_suffix(prefix):
    random_suffix = str(uuid.uuid4())[:8]
    bucket_name = f"{prefix.lower()}-{random_suffix.lower()}"
    try:
        if region == "us-east-1":
            # For us-east-1, we don't specify LocationConstraint
            response = s3_client.create_bucket(
                Bucket=bucket_name
            )
        else:
            # For other regions, we need to specify LocationConstraint
            response = s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': region
                }
            )

        print(f"Successfully created bucket: {bucket_name}")

        # Wait for the bucket to be available
        waiter = s3_client.get_waiter('bucket_exists')
        waiter.wait(Bucket=bucket_name)
        return bucket_name

    except Exception as e:
        print(f"Error creating bucket: {e}")
        return None

def upload_directory(path, bucket_name):
    for root,dirs,files in os.walk(path):
        for file in files:
            file_to_upload = os.path.join(root,file)
            basename = os.path.basename(file_to_upload)
            if basename == ".DS_Store":
                continue
            print(f"uploading file {file_to_upload} to {bucket_name}")
            s3_client.upload_file(file_to_upload,bucket_name,file)

def create_bedrock_knowledge_base(name, description, s3_bucket):
    knowledge_base = BedrockKnowledgeBase(
        kb_name=name,
        kb_description=description,
        data_bucket_name=s3_bucket,
        embedding_model = "amazon.titan-embed-text-v2:0"
    )
    
    # Wait for Knowledge Base to become ACTIVE
    kb_id = knowledge_base.get_knowledge_base_id()
    print(f"Waiting for Knowledge Base {kb_id} to become ACTIVE...")
    max_wait = 120  # 2 minutes
    waited = 0
    while waited < max_wait:
        response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
        status = response['knowledgeBase']['status']
        print(f"Knowledge Base status: {status}")
        if status == 'ACTIVE':
            print("Knowledge Base is ACTIVE")
            break
        elif status == 'FAILED':
            print(f"Knowledge Base creation FAILED: {response['knowledgeBase'].get('failureReasons', [])}")
            raise Exception("Knowledge Base creation failed")
        time.sleep(10)
        waited += 10
    
    if waited >= max_wait:
        print("Warning: Knowledge Base did not become ACTIVE within timeout")
    
    return knowledge_base

def troubleshoot_kb_setup(knowledge_base_id, s3_bucket, role_arn):
    """Check common issues that cause ingestion failures"""
    print(f"\n{'='*80}")
    print("Troubleshooting Knowledge Base Setup")
    print(f"{'='*80}")
    
    # 1. Check if files exist in S3
    print("\n1. Checking S3 bucket contents...")
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket)
        if 'Contents' in response:
            print(f"   ✓ Found {len(response['Contents'])} files in bucket")
            for obj in response['Contents']:
                print(f"     - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print(f"   ✗ No files found in bucket {s3_bucket}")
    except Exception as e:
        print(f"   ✗ Error accessing S3: {e}")
    
    # 2. Check IAM role
    print("\n2. Checking IAM role...")
    role_name = role_arn.split('/')[-1]
    try:
        iam_client = boto3.client('iam', region_name=region)
        role = iam_client.get_role(RoleName=role_name)
        print(f"   ✓ Role exists: {role_name}")
        
        # Check attached policies
        policies = iam_client.list_attached_role_policies(RoleName=role_name)
        print(f"   Attached policies: {len(policies['AttachedPolicies'])}")
        for policy in policies['AttachedPolicies']:
            print(f"     - {policy['PolicyName']}")
    except Exception as e:
        print(f"   ✗ Error checking IAM role: {e}")
    
    # 3. Check Knowledge Base status
    print("\n3. Checking Knowledge Base status...")
    try:
        kb_response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
        kb_status = kb_response['knowledgeBase']['status']
        print(f"   Knowledge Base status: {kb_status}")
        if kb_status != 'ACTIVE':
            print(f"   ✗ Knowledge Base is not ACTIVE")
            if 'failureReasons' in kb_response['knowledgeBase']:
                print(f"   Failure reasons: {kb_response['knowledgeBase']['failureReasons']}")
    except Exception as e:
        print(f"   ✗ Error checking KB: {e}")
    
    print(f"\n{'='*80}\n")

def ingest_single_document(knowledge_base_id, data_source_id, s3_uri, doc_id, metadata):
    """Ingest a single document and monitor its status"""
    
    # Try without metadata first to isolate the issue
    doc = {
        'content': {
            'custom': {
                'customDocumentIdentifier': {'id': doc_id},
                's3Location': {'uri': s3_uri},
                'sourceType': 'S3_LOCATION'
            },
            'dataSourceType': 'CUSTOM'
        }
    }
    
    print(f'\nIngesting: {s3_uri} -> ID: "{doc_id}"')
    print(f"Document structure: {json.dumps(doc, indent=2)}")
    
    try:
        response = bedrock_agent_client.ingest_knowledge_base_documents(
            dataSourceId=data_source_id,
            documents=[doc],
            knowledgeBaseId=knowledge_base_id
        )
        
        doc_detail = response['documentDetails'][0]
        print(f"Initial status: {doc_detail['status']}")
        
        # Monitor status
        max_wait = 60
        waited = 0
        while waited < max_wait:
            time.sleep(5)
            waited += 5
            
            # Get document status
            try:
                get_response = bedrock_agent_client.get_knowledge_base_documents(
                    knowledgeBaseId=knowledge_base_id,
                    dataSourceId=data_source_id,
                    documentIdentifiers=[{'dataSourceType': 'CUSTOM', 'custom': {'id': doc_id}}]
                )
                
                if get_response['documentDetails']:
                    status = get_response['documentDetails'][0]['status']
                    print(f"Status after {waited}s: {status}")
                    
                    if status == 'INDEXED':
                        print(f"✓ Successfully indexed: {doc_id}")
                        return True
                    elif status == 'FAILED':
                        print(f"✗ Failed to index: {doc_id}")
                        doc_details = get_response['documentDetails'][0]
                        print(f"Full document details: {json.dumps(doc_details, indent=2, default=str)}")
                        return False
            except Exception as e:
                print(f"Error checking status: {e}")
        
        print(f"⚠ Timeout waiting for {doc_id} to index")
        return False
        
    except Exception as e:
        print(f'✗ Exception ingesting {doc_id}: {e}')
        return False

def ingest_knowledge_base_documents(knowledge_base_id, data_source_id, s3_bucket, kb_folder):
    # Metadata mapping - fixed to match actual filenames
    metadata_map = {
        'cat-food-wikipedia.pdf': {'animal': ['cat'], 'topic': ['food']},
        'cat-play-and-toys-wikipedia.pdf': {'animal': ['cat'], 'topic': ['play', 'toys']},
        'dog-food-wikipedia.pdf': {'animal': ['dog'], 'topic': ['food']},
        'dog-grooming-wikipedia.pdf': {'animal': ['dog'], 'topic': ['grooming']}
    }
    
    kb_files = [file for file in os.listdir(kb_folder) if file.endswith('.pdf')]
    
    success_count = 0
    fail_count = 0
    
    for kb_file in kb_files:
        s3_uri = f's3://{s3_bucket}/{kb_file}'
        clean_filename = re.sub(r'[\s-]+', '-', kb_file)
        doc_id = os.path.splitext(clean_filename)[0].lower()
        metadata = metadata_map.get(kb_file, {})
        
        if ingest_single_document(knowledge_base_id, data_source_id, s3_uri, doc_id, metadata):
            success_count += 1
        else:
            fail_count += 1
            print(f"\n⚠ Stopping ingestion due to failure. Fix the issue before continuing.")
            break
    
    print(f"\n{'='*80}")
    print(f"Ingestion Summary: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*80}")
    
    return success_count, fail_count


def main():
    folder = 'pets-kb-files'
    if not os.path.isdir(folder):
        download_file('https://d2t0q66iga6m3s.cloudfront.net/pets-kb-files.zip')
        extract_zip_file('pets-kb-files.zip')
        s3_bucket = create_s3_bucket_with_random_suffix('bedrock-kb-bucket')
        print(f'Created S3 bucket: {s3_bucket}')
        upload_directory("pets-kb-files", s3_bucket)
    else:
        print('Skipping download as folder {folder} already exists.')
        buckets = s3_client.list_buckets()['Buckets']
        s3_buckets = [ b['Name'] for b in buckets if b['Name'].startswith('bedrock-kb-bucket') ]
        s3_bucket = s3_buckets[0]

    # Create Bedrock Knowledge Base
    response = bedrock_agent_client.list_knowledge_bases()
    knowledge_bases = response.get('knowledgeBaseSummaries')
    if not len(knowledge_bases):
        random_suffix = str(uuid.uuid4())[:8]
        knowledge_base = create_bedrock_knowledge_base(
            name = f'pets-kb-s3vectordb-{random_suffix}',
            description = 'Pets Knowledge Base on cats and dogs',
            s3_bucket = s3_bucket
        )
        knowledge_base_id = knowledge_base.get_knowledge_base_id()
        data_source_id = knowledge_base.get_datasource_id()
        print(f'Created Bedrock Knowledge Base with ID: {knowledge_base_id}')
    else:
        print('Skipping Bedrock Knowledge Base Creation')
        knowledge_base_id = knowledge_bases[0]['knowledgeBaseId']
        response = bedrock_agent_client.list_data_sources(knowledgeBaseId=knowledge_base_id)
        data_sources = response['dataSourceSummaries']
        data_source_ids = [ d['dataSourceId'] for d in data_sources ]
        if len(data_source_ids):
            data_source_id = data_source_ids[0]
        else:
            print('Error: Data source not created. Please create a custom data source manually')
            return

    # Wait for Data Source to be ready
    print(f"Waiting for Data Source {data_source_id} to be ready...")
    time.sleep(10)
    
    # Troubleshoot setup before ingestion
    kb_response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
    role_arn = kb_response['knowledgeBase']['roleArn']
    troubleshoot_kb_setup(knowledge_base_id, s3_bucket, role_arn)
    
    print(f'Loading documents into Bedrock Knowledge Base: {knowledge_base_id}')
    print(f'Data Source ID: {data_source_id}')

    # Ingest documents from S3 into Bedrock Knowledge Base
    # Requires the appropriate S3 bucket permissions in the
    # Knowledge Base role: AmazonBedrockExecutionRoleForKnowledgeBase_xx 
    ingest_knowledge_base_documents(
        knowledge_base_id = knowledge_base_id,
        data_source_id = data_source_id,
        s3_bucket = s3_bucket,
        kb_folder = folder
    )

if __name__ == '__main__':
    main()
