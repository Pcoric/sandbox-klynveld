import os
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.steps import ParallelRunStep, ParallelRunConfig
from azureml.data.data_reference import DataReference
from azureml.data.datapath import DataPath
from azureml.core import Experiment
from azureml.pipeline.core import PipelineEndpoint
from azureml.pipeline.core import PipelineDraft
import http.client, urllib.request, urllib.parse, urllib.error, base64
import requests, time, json
import numpy as np

#from util.attach_compute import get_compute
#from util.env_variables import Env
#from util.manage_environment import get_environment
#from sklearn.datasets import load_tchmielMLOps
#import pandas as pd

from util.aml_interface import AMLInterface

def create_aml_environment(aml_workspace, aml_env_name):
    aml_env = Environment(name=aml_env_name)
    conda_dep = CondaDependencies()
    conda_dep.add_pip_package("numpy==1.18.2")
    conda_dep.add_pip_package("pandas==1.0.3")
    conda_dep.add_pip_package("scipy==1.0.0")
    conda_dep.add_pip_package("pytest==4.3.0")
    conda_dep.add_pip_package("scikit-learn==0.22.2.post1")
    conda_dep.add_pip_package("joblib==0.14.1")
    conda_dep.add_pip_package('PyPDF2')
    conda_dep.add_pip_package('pdf2image') # Replacement for wand package
    conda_dep.add_pip_package('pillow')
    conda_dep.add_channel('conda-forge')
    conda_dep.add_conda_package('poppler') # dependency for pdf2image package
    '''
    whl_filepath = retrieve_whl_filepath()
    whl_url = Environment.add_private_pip_wheel(
        workspace=aml_workspace,
        file_path=whl_filepath,
        exist_ok=True
    )
    conda_dep.add_pip_package(whl_url)
    '''
    aml_env.python.conda_dependencies = conda_dep
    aml_env.docker.enabled = True
    return aml_env, conda_dep

def get_compute(aml_workspace, cluster_name):
    try:
        # Creating an object from already provisioned compute target
        cluster = AmlCompute(aml_workspace, cluster_name)
        print("found existing cluster: ", cluster_name)
    except ComputeTargetException:
        # Provisioning a new compute target  
        print("No Cluster Found")
    print("cluster attached: ", cluster_name)
    return cluster


def main():
    # variables in use in this script
    '''
        TENANT_ID
        SPN_ID
        SPN_PASSWORD
        AML_WORKSPACE_NAME
        AML_ENVIRONMENT_NAME
        RESOURCE_GROUP
        COMPUTE_NAME
        SUBSCRIPTION_ID
    '''
    # Retrieve vars from env (Passed through from azure-ml-pipeline.yml)
    workspace_name = os.environ['AML_WORKSPACE_NAME']
    resource_group = os.environ['RESOURCE_GROUP']
    subscription_id = os.environ['SUBSCRIPTION_ID']
    compute_name = os.environ['COMPUTE_NAME']
    aml_environment_name = os.environ['AML_ENVIRONMENT_NAME']
    blob_datastore_name=os.environ['BLOBSTORENAME']
    account_name=os.environ['DATASETDS']
    container_name=os.environ['CONTNAME']
    account_key = os.environ['ACCTKEY']


    # Get Azure machine learning workspace, interactive Auth
        #TO-DO: Add service principal to az cli
    aml_workspace = Workspace.get(name = workspace_name, subscription_id = subscription_id, resource_group = resource_group)
    print("get_workspace:")
    print(aml_workspace)

    #aml_interface = AMLInterface(aml_workspace, subscription_id, workspace_name, resource_group)

    # Get Azure machine learning cluster
    aml_compute = get_compute(aml_workspace, compute_name)
    #if aml_compute is not None:
    #    print("aml_compute:")
    #    print(aml_compute)

    # Create aml env
    aml_env, conda_dep = create_aml_environment(aml_workspace, aml_environment_name)

    # Register AML env
    aml_env.register(workspace=aml_workspace)
    
    # Register Datastore
        # TO-DO: Build data pipeline to populate data
    Datastore.register_azure_blob_container(
            workspace=aml_workspace,
            datastore_name=blob_datastore_name,
            container_name=container_name,
            account_name=account_name,
            account_key=account_key
        )
    # Create runconfig from conda dependencies
    runconfig = RunConfiguration(conda_dependencies=conda_dep)
    runconfig.environment.environment_variables["DATASTORE_NAME"] = blob_datastore_name  # NOQA: E501
    
    # Get datastore as object
    ds = Datastore.get(aml_workspace, blob_datastore_name)




def init():
    step_main = 's200_sheet_classifier'
    print('Running step: {}'.format(step_main))

def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    print("Mini Batch: {}".format(mini_batch))
    headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Prediction-Key': '7129945c0e714b428e60cef9f90c56ad',
    }
    
    endpoint_url = "https://eastus2.api.cognitive.microsoft.com/customvision/v3.0/Prediction/dcc66f32-f086-4609-b099-d71743ee97a7/classify/iterations/Iteration12/image"
    resultList = []

    step_name = "sheet_classifier"

    for file in os.listdir(mini_batch):
        
        basename = os.path.basename(file)
        fullfile = mini_batch + file
        data = open(fullfile,"rb").read()
        response = requests.post(endpoint_url, headers=headers, data=data)
        
        root_dict = {}
        root_dict = response.json()
        root_dict["name"] = basename
        root_dict["step_name"] = step_name
        
        if response.status_code != 200:
            root_dict['error'] = {}
            root_dict['error']['status_code'] =  response.status_code
            best_result = 'error'
            #best_result = response.status_code
            #print("request failed for file: ", file)
            #print("response.json", response.json)
        else:
            #print (response.json)
            predictions = response.json()["predictions"]
            prob = [x["probability"] for x in predictions]
            prob_class = [x["tagName"] for x in predictions]
            best_result = prob_class[np.argmax(prob)]

        root_dict['result'] = {}
        root_dict['result']['sheet_type'] = best_result
            
        resultList.append(json.dumps(root_dict))

    return resultList


if __name__ == "__main__":
    # Test scoring
    init()
    # Maybe not necessary, since i'm not passing arguments (yet)
    parser = argparse.ArgumentParser(description="Start a tensorflow model serving")
    parser.add_argument('--input', type=str, dest="blob_input_data", required=True)
    parser.add_argument('--output', type=str, dest="output_data1", required=True)
    args = parser.parse_args()
    # Need to replace with image data, but it's been passed. (for local we'll call out to the images folder, and )
    test_images = os.path.join(args.blob_input_data,"")
    prediction = run(test_images)
    
    # make output dirs, if needed
    os.makedirs(args.output_data1, exist_ok=True)
    # Are we passing this a folder or multiple images???
    print("Test result: ", prediction)