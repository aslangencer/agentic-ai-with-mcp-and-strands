#!/usr/bin/env python

# S3-based Knowledge Base (simplified version without OpenSearch Serverless)

import json
import boto3
import time
from botocore.exceptions import ClientError
import pprint
from retrying import retry

valid_embedding_models = [
    "cohere.embed-multilingual-v3",
    "cohere.embed-english-v3",
    "amazon.titan-embed-text-v2:0"
]
pp = pprint.PrettyPrinter(indent=2)


def interactive_sleep(seconds: int):
    """
    Support functionality to induce an artificial 'sleep' to the code in order to wait for resources to be available
    Args:
        seconds (int): number of seconds to sleep for
    """
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)


class BedrockKnowledgeBase:
    """
    Support class that allows for:
        - creation (or retrieval) of a Knowledge Base for Amazon Bedrock with S3 vector storage
        - Ingestion of data into the Knowledge Base
        - Deletion of all resources created
    """
    def __init__(
            self,
            kb_name,
            kb_description=None,
            data_bucket_name=None,
            embedding_model="amazon.titan-embed-text-v2:0"
    ):
        """
        Class initializer
        Args:
            kb_name (str): the knowledge base name
            kb_description (str): knowledge base description
            data_bucket_name (str): name of s3 bucket to connect with knowledge base
            embedding_model (str): embedding model to use
        """
        import os
        boto3_session = boto3.session.Session()
        self.region_name = os.getenv('AWS_REGION') or boto3_session.region_name
        self.iam_client = boto3_session.client('iam', region_name=self.region_name)
        self.account_number = boto3.client('sts', region_name=self.region_name).get_caller_identity().get('Account')
        self.suffix = str(self.account_number)[:4]
        self.s3_client = boto3.client('s3', region_name=self.region_name)
        self.s3vectors_client = boto3.client('s3vectors', region_name=self.region_name)
        self.bedrock_agent_client = boto3.client('bedrock-agent', region_name=self.region_name)

        self.kb_name = kb_name
        self.kb_description = kb_description
        if data_bucket_name is not None:
            self.bucket_name = data_bucket_name
        else:
            self.bucket_name = f"{self.kb_name}-{self.suffix}"
        if embedding_model not in valid_embedding_models:
            valid_embeddings_str = str(valid_embedding_models)
            raise ValueError(f"Invalid embedding model. Your embedding model should be one of {valid_embeddings_str}")
        self.embedding_model = embedding_model
        self.kb_execution_role_name = f'AmazonBedrockExecutionRoleForKnowledgeBase_{self.suffix}'
        self.fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{self.suffix}'
        self.s3_policy_name = f'AmazonBedrockS3PolicyForKnowledgeBase_{self.suffix}'

        print("========================================================================================")
        print(f"Step 1 - Creating or retrieving {self.bucket_name} S3 bucket for Knowledge Base")
        self.create_s3_bucket()
        print("========================================================================================")
        print(f"Step 2 - Creating S3 Vector bucket and index")
        self.vector_bucket_name, self.vector_index_name, self.vector_bucket_arn, self.vector_index_arn = self.create_s3_vector_resources()
        print("========================================================================================")
        print(f"Step 3 - Creating Knowledge Base Execution Role ({self.kb_execution_role_name}) and Policies")
        self.bedrock_kb_execution_role = self.create_bedrock_kb_execution_role()
        print("========================================================================================")
        print(f"Step 4 - Creating Knowledge Base with S3 vector storage")
        self.knowledge_base, self.data_source = self.create_knowledge_base()
        print("========================================================================================")

    def create_s3_bucket(self):
        """
        Check if bucket exists, and if not create S3 bucket for knowledge base data source
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f'Bucket {self.bucket_name} already exists - retrieving it!')
        except ClientError as e:
            print(f'Creating bucket {self.bucket_name}')
            if self.region_name == "us-east-1":
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name
                )
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region_name}
                )

    def create_s3_vector_resources(self):
        """
        Create S3 Vector bucket and index for Knowledge Base vector storage
        """
        vector_bucket_name = f"{self.kb_name}-vectors-{self.suffix}".lower()
        vector_index_name = f"{self.kb_name}-index".lower()
        
        # Create vector bucket
        try:
            response = self.s3vectors_client.create_vector_bucket(
                vectorBucketName=vector_bucket_name
            )
            vector_bucket_arn = response['vectorBucketArn']
            print(f"Created S3 vector bucket: {vector_bucket_name}")
        except self.s3vectors_client.exceptions.ConflictException:
            response = self.s3vectors_client.get_vector_bucket(vectorBucketName=vector_bucket_name)
            vector_bucket_arn = response['vectorBucketArn']
            print(f"S3 vector bucket already exists: {vector_bucket_name}")
        
        # Create vector index
        try:
            response = self.s3vectors_client.create_index(
                vectorBucketName=vector_bucket_name,
                indexName=vector_index_name,
                dimension=1024,  # For amazon.titan-embed-text-v2:0
                dataType='float32',
                distanceMetric='euclidean'
            )
            vector_index_arn = response['indexArn']
            print(f"Created S3 vector index: {vector_index_name}")
        except self.s3vectors_client.exceptions.ConflictException:
            response = self.s3vectors_client.get_index(
                vectorBucketName=vector_bucket_name,
                indexName=vector_index_name
            )
            vector_index_arn = response['indexArn']
            print(f"S3 vector index already exists: {vector_index_name}")
        
        return vector_bucket_name, vector_index_name, vector_bucket_arn, vector_index_arn

    def create_bedrock_kb_execution_role(self):
        """
        Create Knowledge Base Execution IAM Role and its required policies.
        If role and/or policies already exist, retrieve them
        Returns:
            IAM role
        """
        foundation_model_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "bedrock:*",
                    "Resource": "*"
                }
            ]
        }

        s3_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "s3:*",
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": "s3vectors:*",
                    "Resource": "*"
                }
            ]
        }

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        try:
            # create policies based on the policy documents
            fm_policy = self.iam_client.create_policy(
                PolicyName=self.fm_policy_name,
                PolicyDocument=json.dumps(foundation_model_policy_document),
                Description='Policy for accessing foundation model',
            )
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            fm_policy = self.iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{self.account_number}:policy/{self.fm_policy_name}"
            )

        try:
            s3_policy = self.iam_client.create_policy(
                PolicyName=self.s3_policy_name,
                PolicyDocument=json.dumps(s3_policy_document),
                Description='Policy for reading documents from s3')
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            s3_policy = self.iam_client.get_policy(
                PolicyArn=f"arn:aws:iam::{self.account_number}:policy/{self.s3_policy_name}"
            )
            # Update the policy with new vector resource ARNs
            self.iam_client.create_policy_version(
                PolicyArn=s3_policy['Policy']['Arn'],
                PolicyDocument=json.dumps(s3_policy_document),
                SetAsDefault=True
            )
        # create bedrock execution role
        try:
            bedrock_kb_execution_role = self.iam_client.create_role(
                RoleName=self.kb_execution_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description='Amazon Bedrock Knowledge Base Execution Role for accessing S3',
                MaxSessionDuration=3600
            )
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            bedrock_kb_execution_role = self.iam_client.get_role(
                RoleName=self.kb_execution_role_name
            )
            # Update trust policy for existing role
            self.iam_client.update_assume_role_policy(
                RoleName=self.kb_execution_role_name,
                PolicyDocument=json.dumps(assume_role_policy_document)
            )
        # fetch arn of the policies and role created above
        s3_policy_arn = s3_policy["Policy"]["Arn"]
        fm_policy_arn = fm_policy["Policy"]["Arn"]

        # attach policies to Amazon Bedrock execution role
        self.iam_client.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=fm_policy_arn
        )
        self.iam_client.attach_role_policy(
            RoleName=bedrock_kb_execution_role["Role"]["RoleName"],
            PolicyArn=s3_policy_arn
        )
        # Wait for IAM role to propagate
        print("Waiting 60 seconds for IAM role and policies to propagate...")
        interactive_sleep(60)
        return bedrock_kb_execution_role

    @retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
    def create_knowledge_base(self):
        """
        Create Knowledge Base with S3 vector storage and its Data Source. If existent, retrieve
        """
        # Ingest strategy - Use default chunking (300 tokens) by omitting chunkingConfiguration
        # or use FIXED_SIZE for explicit control
        chunking_strategy_configuration = {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 300,
                "overlapPercentage": 20
            }
        }

        # The embedding model used by Bedrock to embed ingested documents, and realtime prompts
        embedding_model_arn = f"arn:aws:bedrock:{self.region_name}::foundation-model/{self.embedding_model}"
        
        print(f"Creating Knowledge Base with role: {self.bedrock_kb_execution_role['Role']['Arn']}")
        print(f"Embedding model: {embedding_model_arn}")
        print(f"Storage type: S3_VECTORS")
        
        try:
            create_kb_response = self.bedrock_agent_client.create_knowledge_base(
                name=self.kb_name,
                description=self.kb_description,
                roleArn=self.bedrock_kb_execution_role['Role']['Arn'],
                knowledgeBaseConfiguration={
                    "type": "VECTOR",
                    "vectorKnowledgeBaseConfiguration": {
                        "embeddingModelArn": embedding_model_arn
                    }
                },
                storageConfiguration={
                    "type": "S3_VECTORS",
                    "s3VectorsConfiguration": {
                        "vectorBucketArn": self.vector_bucket_arn,
                        "indexArn": self.vector_index_arn
                    }
                }
            )
            kb = create_kb_response["knowledgeBase"]
            pp.pprint(kb)
        except Exception as e:
            print(f"\nError creating knowledge base: {e}")
            print(f"\nRole ARN being used: {self.bedrock_kb_execution_role['Role']['Arn']}")
            print(f"\nChecking role details...")
            role_details = self.iam_client.get_role(RoleName=self.kb_execution_role_name)
            print(f"\nRole Trust Policy:")
            pp.pprint(role_details['Role']['AssumeRolePolicyDocument'])
            print(f"\nAttached Policies:")
            attached_policies = self.iam_client.list_attached_role_policies(RoleName=self.kb_execution_role_name)
            pp.pprint(attached_policies)
            raise
        except self.bedrock_agent_client.exceptions.ConflictException:
            kbs = self.bedrock_agent_client.list_knowledge_bases(
                maxResults=100
            )
            kb_id = None
            for kb in kbs['knowledgeBaseSummaries']:
                if kb['name'] == self.kb_name:
                    kb_id = kb['knowledgeBaseId']
            response = self.bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)
            kb = response['knowledgeBase']
            pp.pprint(kb)

        # Create a DataSource in KnowledgeBase
        try:
            create_ds_response = self.bedrock_agent_client.create_data_source(
                name=self.kb_name,
                description=self.kb_description,
                knowledgeBaseId=kb['knowledgeBaseId'],
                dataSourceConfiguration={
                    "type": "CUSTOM"
                },
                vectorIngestionConfiguration={
                    "chunkingConfiguration": chunking_strategy_configuration
                }
            )
            ds = create_ds_response["dataSource"]
            pp.pprint(ds)
        except self.bedrock_agent_client.exceptions.ConflictException:
            ds_id = self.bedrock_agent_client.list_data_sources(
                knowledgeBaseId=kb['knowledgeBaseId'],
                maxResults=100
            )['dataSourceSummaries'][0]['dataSourceId']
            get_ds_response = self.bedrock_agent_client.get_data_source(
                dataSourceId=ds_id,
                knowledgeBaseId=kb['knowledgeBaseId']
            )
            ds = get_ds_response["dataSource"]
            pp.pprint(ds)
        return kb, ds

    def start_ingestion_job(self):
        """
        Start an ingestion job to synchronize data from an S3 bucket to the Knowledge Base
        """
        # Start an ingestion job
        start_job_response = self.bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
            dataSourceId=self.data_source["dataSourceId"]
        )
        job = start_job_response["ingestionJob"]
        pp.pprint(job)
        # Get job
        while job['status'] != 'COMPLETE':
            get_job_response = self.bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=self.knowledge_base['knowledgeBaseId'],
                dataSourceId=self.data_source["dataSourceId"],
                ingestionJobId=job["ingestionJobId"]
            )
            job = get_job_response["ingestionJob"]
        pp.pprint(job)
        interactive_sleep(40)

    def get_knowledge_base_id(self):
        """
        Get Knowledge Base Id
        """
        pp.pprint(self.knowledge_base["knowledgeBaseId"])
        return self.knowledge_base["knowledgeBaseId"]

    def get_datasource_id(self):
        """
        Get Data Source Id
        """
        pp.pprint(self.data_source["dataSourceId"])
        return self.data_source["dataSourceId"]

    def get_bucket_name(self):
        """
        Get the name of the bucket connected with the Knowledge Base Data Source
        """
        pp.pprint(f"Bucket connected with KB: {self.bucket_name}")
        return self.bucket_name

    def delete_kb(self, delete_s3_bucket=False, delete_iam_roles_and_policies=True):
        """
        Delete the Knowledge Base resources
        Args:
            delete_s3_bucket (bool): boolean to indicate if s3 bucket should also be deleted
            delete_iam_roles_and_policies (bool): boolean to indicate if IAM roles and Policies should also be deleted
        """
        self.bedrock_agent_client.delete_data_source(
            dataSourceId=self.data_source["dataSourceId"],
            knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
        )
        self.bedrock_agent_client.delete_knowledge_base(
            knowledgeBaseId=self.knowledge_base['knowledgeBaseId']
        )
        if delete_s3_bucket:
            self.delete_s3()
        if delete_iam_roles_and_policies:
            self.delete_iam_roles_and_policies()

    def delete_iam_roles_and_policies(self):
        """
        Delete IAM Roles and policies used by the Knowledge Base
        """
        fm_policy_arn = f"arn:aws:iam::{self.account_number}:policy/{self.fm_policy_name}"
        s3_policy_arn = f"arn:aws:iam::{self.account_number}:policy/{self.s3_policy_name}"
        self.iam_client.detach_role_policy(
            RoleName=self.kb_execution_role_name,
            PolicyArn=s3_policy_arn
        )
        self.iam_client.detach_role_policy(
            RoleName=self.kb_execution_role_name,
            PolicyArn=fm_policy_arn
        )
        self.iam_client.delete_role(RoleName=self.kb_execution_role_name)
        self.iam_client.delete_policy(PolicyArn=s3_policy_arn)
        self.iam_client.delete_policy(PolicyArn=fm_policy_arn)
        return 0

    def delete_s3(self):
        """
        Delete the objects contained in the Knowledge Base S3 bucket.
        Once the bucket is empty, delete the bucket
        """
        objects = self.s3_client.list_objects(Bucket=self.bucket_name)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
        self.s3_client.delete_bucket(Bucket=self.bucket_name)
