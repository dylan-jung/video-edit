import pulumi
import pulumi_gcp as gcp
import os

# Config
config = pulumi.Config()
gcp_config = pulumi.Config("gcp")
project_id = gcp_config.require("project")
region = gcp_config.get("region", "asia-northeast3")
indexer_image = config.get("indexer_image", f"gcr.io/{project_id}/video-indexer:latest")

# 1. Cloud Storage Bucket (Upload & Result)
bucket = gcp.storage.Bucket("video-assets-bucket",
    location=region,
    uniform_bucket_level_access=True,
    force_destroy=True
)

# 2. Pub/Sub Topic
topic = gcp.pubsub.Topic("indexing-jobs-topic",
    name="indexing-jobs"
)

# 3. Service Accounts & IAM

# 3-1. Cloud Run Worker Service Account
worker_sa = gcp.serviceaccount.Account("indexer-worker-sa",
    account_id="indexer-worker-sa",
    display_name="Indexer Cloud Run Worker SA"
)

# Worker needs access to Storage (Read/Write)
worker_storage_binding = gcp.projects.IAMMember("worker-storage-binding",
    project=project_id,
    role="roles/storage.admin", # Or roles/storage.objectAdmin
    member=pulumi.Output.concat("serviceAccount:", worker_sa.email)
)

# 3-2. Cloud Function Trigger Service Account
trigger_sa = gcp.serviceaccount.Account("trigger-sa",
    account_id="gcs-trigger-sa",
    display_name="GCS Trigger Cloud Function SA"
)

# Trigger needs access to Publish to Pub/Sub
trigger_pubsub_binding = gcp.pubsub.TopicIAMMember("trigger-pub_sub-binding",
    topic=topic.name,
    role="roles/pubsub.publisher",
    member=pulumi.Output.concat("serviceAccount:", trigger_sa.email)
)

# 3-3. Pub/Sub Subscription Invoker Service Account (Push Identity)
invoker_sa = gcp.serviceaccount.Account("pubsub-invoker-sa",
    account_id="pubsub-invoker-sa",
    display_name="Pub/Sub Push Subscription Invoker SA"
)

# 4. Cloud Run Service (The Worker)
indexer_service = gcp.cloudrunv2.Service("indexer-service",
    location=region,
    template=gcp.cloudrunv2.ServiceTemplateArgs(
        service_account=worker_sa.email,
        containers=[gcp.cloudrunv2.ServiceTemplateContainerArgs(
            image=indexer_image,
            envs=[
                gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                    name="GCP_PROJECT_ID",
                    value=project_id
                ),
                gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                    name="GCOM_BUCKET_NAME", # Assuming code might need this
                    value=bucket.name
                )
            ],
            resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                limits={"cpu": "2", "memory": "4Gi"} # Heavy task needs resources
            )
        )]
    ),
    traffics=[gcp.cloudrunv2.ServiceTrafficArgs(
        type="TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST",
        percent=100
    )]
)

# Allow Invoker SA to invoke Cloud Run
invoker_run_binding = gcp.cloudrunv2.ServiceIAMMember("invoker-run-binding",
    project=project_id,
    location=region,
    name=indexer_service.name,
    role="roles/run.invoker",
    member=pulumi.Output.concat("serviceAccount:", invoker_sa.email)
)

# 5. Pub/Sub Subscription (Push to Cloud Run)
subscription = gcp.pubsub.Subscription("indexing-subscription",
    topic=topic.name,
    push_config=gcp.pubsub.SubscriptionPushConfigArgs(
        push_endpoint=indexer_service.uri.apply(lambda uri: f"{uri}/pubsub/push"),
        oidc_token=gcp.pubsub.SubscriptionPushConfigOidcTokenArgs(
            service_account_email=invoker_sa.email
        )
    ),
    ack_deadline_seconds=600 # 10 minutes max for indexing
)

# 6. Cloud Function (The Trigger)
# Archive the source code
# Correct path relative to where `pulumi up` is run (likely root of project or infra folder?)
# Assuming `pulumi up` is run from `infrastructure/` directory, the path is `../src/gcp/functions/storage_trigger`
source_archive_object = gcp.storage.BucketObject("trigger-source-code",
    bucket=bucket.name,
    source=pulumi.FileArchive("../src/gcp/functions/storage_trigger"),
    name="source_code/trigger_function.zip" # Unique name or hash usually better
)

trigger_function = gcp.cloudfunctions.Function("gcs-trigger-function",
    region=region,
    runtime="python310",
    source_archive_bucket=bucket.name,
    source_archive_object=source_archive_object.name,
    entry_point="gcs_object_finalized",
    trigger_http=False,
    event_trigger=gcp.cloudfunctions.FunctionEventTriggerArgs(
        event_type="google.storage.object.finalize",
        resource=bucket.name
    ),
    service_account_email=trigger_sa.email,
    environment_variables={
        "GCP_PROJECT_ID": project_id,
        "PUBSUB_TOPIC_ID": topic.name
    }
)

# Exports
pulumi.export("bucket_name", bucket.name)
pulumi.export("topic_name", topic.name)
pulumi.export("cloud_run_url", indexer_service.uri)
pulumi.export("function_name", trigger_function.name)
