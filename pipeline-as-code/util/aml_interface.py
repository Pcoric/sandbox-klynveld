from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication

class AMLInterface:
    def __init__(self, spn_credentials, subscription_id,
                 workspace_name, resource_group):
        auth = ServicePrincipalAuthentication(
            **spn_credentials
        )
        self.workspace = Workspace(
            workspace_name=workspace_name,
            auth=auth,
            subscription_id=subscription_id,
            resource_group=resource_group
        )
    
    def register_aml_environment(self, environment):
        environment.register(workspace=self.workspace)
    
    def register_datastore(self, datastore_name, blob_container,
                           storage_acct_name, storage_acct_key):
        Datastore.register_azure_blob_container(
            workspace=self.workspace,
            datastore_name=datastore_name,
            container_name=blob_container,
            account_name=storage_acct_name,
            account_key=storage_acct_key
        )