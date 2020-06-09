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


    # Specify output data for object
    output_data1 = PipelineData(
        "output_data1",
        datastore=ds,
        output_name="output_data1")

    named_ds = "sampleds"
    # copied from: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.filedatasetfactory?view=azure-ml-py#from-files-path--validate-true-

    # Create data store path for file dataset creation
    datastore_path = DataPath(ds, 'aiml20/testimages/*.jpg')
    # Create file dataset
    file_dataset = Dataset.File.from_files(path=datastore_path)
    # Register file dataset with workspace
    registered_flds = file_dataset.register(aml_workspace, named_ds, create_new_version=True)
    # Create named input object to pass run
    named_flds = registered_flds.as_named_input(named_ds)

    # Create output file name
    s200_prs_filename = "sheet_classifer.txt"
    # Parallel run config including aml_env, score.py file and compute target
    s200_parallel_run_config = ParallelRunConfig(
                    environment=aml_env,
                    entry_script="score.py",
                    output_action='append_row',
                    mini_batch_size="1",
                    error_threshold=1,
                    source_directory="./pipeline-as-code/src/",
                    compute_target=aml_compute, 
                    append_row_file_name= s200_prs_filename,
                    node_count=3)
    
    # Create Parallel Run Step Object
    image_classifier = ParallelRunStep(
        name="parallelrunstep",
        arguments=[],
        inputs=[named_flds],
        output=output_data1,
        parallel_run_config=s200_parallel_run_config,
        allow_reuse=True #[optional - default value True]
    )

    # Establish Pipeline Tags
    ptags = {'dev': 'true'}
    # Aggregate the steps, in this example there's only one
    simpleModel = [image_classifier]
    # Create a Pipeline Object, note this won't save anything. The Pipeline Object must be published or saved as a draft. 
    pipeline = Pipeline(workspace=aml_workspace, steps=[simpleModel])
    # Register the Pipeline as a Draft and add tags/properties such as customer, environment etc. The tags are immutable the properties 
    pipeline_draft = PipelineDraft.create(workspace=aml_workspace, name="minihackDraft", description="showing that pipeline draft capability helps monitor", experiment_name='minihack', pipeline=pipeline, tags=ptags, properties={'customer':'bentley'})
    pipeline.validate()
    # Submit a pipeline run through the experiment,a submitted run can be a ScriptRunConfig, AutoMLConfig, Pipeline, Published Pipeline or PipelineEndpoint
    pipeline_run = Experiment(aml_workspace, 'minihack').submit(pipeline, tags=ptags)
    pipeline_run.wait_for_completion()
    # Users can submit a pipeline_run through a pipeline draft using the submit_run function
    # pipeline_run_example = pipeline_draft_submit_run()

    # Publish a successfully completed pipeline_run.
    published_pipeline = pipeline_run.publish_pipeline(
     name="minihackpipeline",
     description="Published pipeline for MiniHack purpose",
     version="1.0")


    #Create a published pipeline endpoint for accessing the pipeline. 
    pipeline_endpoint = PipelineEndpoint.publish(workspace=aml_workspace, name="PipelineEndpointTest",
                                            pipeline=published_pipeline, description="Test description Notebook")
    # Create a reusable Azure ML environment
    #environment = get_environment(aml_workspace, e.aml_env_name, create_new=e.rebuild_env)  #
    #run_config = RunConfiguration()
    #run_config.environment = environment
    #run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_name  # NOQA: E501
'''
    # Parameters for this specific pipeline
    pp_input_blob_link = PipelineParameter(name="input_blob_link", default_value="null")
'''
    # Pipeline Data objects to pass the files and mounted storages to pass between steps
'''
    output_base_dir = "output"
    pd_output_pngs = PipelineData(output_base_dir + "_pngs", datastore=ds) 
    pd_output_pdfs = PipelineData(output_base_dir + "_pdfs", datastore=ds) 
    pd_output_others = PipelineData(output_base_dir + "_outputs", datastore=ds)
    '''

if __name__ == '__main__':
    main()