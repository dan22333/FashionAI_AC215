from kfp import dsl


# Define a Container Component
@dsl.component(
    base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"]
)
def model_training():
    print("Model Training Job")

    import google.cloud.aiplatform as aip

    # Hardcoded values based on the shell script
    project = "fashion-ai-438801"  # Replace with your GCP project ID
    location = "us-central1"  # Replace with your GCP region
    staging_bucket = "gs://fashionai_training"  # Replace with your staging bucket URI
    gcs_data_bucket_uri = "gs://fashionai_training"  # Replace with your GCS data bucket URI

    # Initialize Vertex AI SDK for Python
    aip.init(project=project, location=location, staging_bucket=staging_bucket)

    # Prebuilt container image and Python package details
    container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-3.py310:latest"
    python_package_gcs_uri = f"{staging_bucket}/trainer.tar.gz"
    python_module_name = "trainer.task"

    # Arguments passed to the training script
    cmdargs = [
        f"--output_dir={gcs_data_bucket_uri}/test_fclip/",
        "--n_samples=100",
        "--bucket_name=fashionai_training"
    ]

    # Machine configuration
    machine_type = "n1-standard-4"
    replica_count = 1

    # Display name for the custom training job
    display_name = "fashionclip_training_job"

    print(python_package_gcs_uri, "python_package_gcs_uri")

    # Create the CustomPythonPackageTrainingJob object
    job = aip.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name=python_module_name,
        container_uri=container_uri,
        project=project,
    )

    # Run the training job
    job.run(
        args=cmdargs,
        replica_count=replica_count,
        machine_type=machine_type,
        #base_output_dir=f"{gcs_data_bucket_uri}",
        sync=True,  # Wait for the job to finish
    )

    print(f"Training job '{display_name}' submitted successfully.")



# Define a Container Component
@dsl.component(
    base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"]
)
def model_deploy(
    bucket_name: str = "",
):
    print("Model Training Job")

    import google.cloud.aiplatform as aip

    # List of prebuilt containers for prediction
    # https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    serving_container_image_uri = (
        "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
    )

    display_name = "Cheese App Model"
    ARTIFACT_URI = f"gs://{bucket_name}/model"

    # Upload and Deploy model to Vertex AI
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_upload
    deployed_model = aip.Model.upload(
        display_name=display_name,
        artifact_uri=ARTIFACT_URI,
        serving_container_image_uri=serving_container_image_uri,
    )
    print("deployed_model:", deployed_model)
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy
    endpoint = deployed_model.deploy(
        deployed_model_display_name=display_name,
        traffic_split={"0": 100},
        machine_type="n1-standard-4",
        accelerator_count=0,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )
    print("endpoint:", endpoint)
