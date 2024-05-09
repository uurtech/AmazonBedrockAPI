# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to stream the response from Anthropic Claude Sonnet (on demand) for a 
multimodal request.
"""

import json
import base64
import logging
import boto3

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def stream_multi_modal_prompt(bedrock_runtime, model_id, input_text, image, max_tokens):
    """
    Streams the response from a multimodal prompt.
    Args:
        bedrock_runtime: The Amazon Bedrock boto3 client.
        model_id (str): The model ID to use.
        input_text (str) : The prompt text
        image (str) : The path to  an image that you want in the prompt.
        max_tokens (int) : The maximum  number of tokens to generate.
    Returns:
        None.
    """

    with open(image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg", "data": encoded_string.decode('utf-8')}}
                ]
            }
        ]
    })

    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, modelId=model_id)

    for event in response.get("body"):
        chunk = json.loads(event["chunk"]["bytes"])

        if chunk['type'] == 'message_delta':
            print(f"\nStop reason: {chunk['delta']['stop_reason']}")
            print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
            print(f"Output tokens: {chunk['usage']['output_tokens']}")

        if chunk['type'] == 'content_block_delta':
            if chunk['delta']['type'] == 'text_delta':
                print(chunk['delta']['text'], end="")


def main():
    """
    Entrypoint for Anthropic Claude Sonnet multimodal prompt example.
    """

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    input_text = "Identify image first and give me fields with their values as json, dont add comment or explaination, give me json only"
    image = "./kimlik.jpeg"
    max_tokens = 100

    try:

        bedrock_runtime = boto3.client('bedrock-runtime')

        stream_multi_modal_prompt(
            bedrock_runtime, model_id, input_text, image, max_tokens)

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))


if __name__ == "__main__":
    main()