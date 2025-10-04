import os
import json
import boto3
import base64


#Lambda 1: Serialize target data from S3
s3 = boto3.client('s3')

def lambda_serialise_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # Read the data and base64-encode it (as a UTF-8 string for JSON)
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        "statusCode": 200,
        "body": {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

#Lambda 2: Invoke classifier
# Fill this in with the name of your deployed model

sm_runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = "image-classification-2025-10-04-06-05-34-974"

def lambda_classifier_handler(event, context):
    # If Step Functions wrapped the payload, unwrap it; otherwise use event directly
    payload = event.get("body", event)

    # Decode base64 image
    image_bytes = base64.b64decode(payload["image_data"])

    # Invoke endpoint (built-in image classifier prefers application/x-image)
    resp = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/x-image",   # if you must, "image/png" also works for PNGs
        Body=image_bytes
    )

    # Keep response as a string (no JSON parsing here)
    result_str = resp["Body"].read().decode("utf-8")

    # Attach to payload for the next step
    payload["inferences"] = result_str

    return {
        "statusCode": 200,
        "body": payload
    }



#Lambda 3: Filter low confidence inferences

# Allow overriding via env var; default to 0.93 as requested
THRESHOLD = 0.84

def lambda_threshold_handler(event, context):
    
    # Unwrap if Step Functions passed {"statusCode":200,"body":{...}}
    payload = event.get("body", event)

    # Grab the inferences from the event (may be a string or a list)
    inferences = payload.get("inferences")

    # Normalize to a Python list of floats
    if isinstance(inferences, str):
        try:
            inferences = json.loads(inferences)        # e.g. "[0.12, 0.88]"
        except Exception:
            # Fallback: try simple split if it's a raw string
            inferences = [float(x) for x in inferences.strip("[] ").split(",")]
    elif isinstance(inferences, dict) and "predictions" in inferences:
        inferences = inferences["predictions"]

    scores = [float(x) for x in inferences]
    meets_threshold = max(scores) >= THRESHOLD

    # If our threshold is met, pass data along; else fail the state
    if not meets_threshold:
        raise Exception(f"THRESHOLD_CONFIDENCE_NOT_MET: max={max(scores):.3f} < {THRESHOLD}")

    return {
        "statusCode": 200,
        "body": json.dumps(payload)
    }