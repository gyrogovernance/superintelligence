# torchrun --nproc-per-node=4 serve.py

import argparse

import uvicorn
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from .api_server import create_api_server
from .inference.gyro import setup_model as gyro_setup_model
from .inference.stub import setup_model as stub_setup_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Responses API server")

    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--inference-backend",
        metavar="BACKEND",
        type=str,
        help="Inference backend to use",
        # default to gyro for GyroSI
        default="gyro",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        help="Path to the GyroSI configuration file",
        default="baby/config.json",
    )
    args = parser.parse_args()

    if args.inference_backend not in ["gyro", "stub"]:
        raise ValueError(f"Invalid inference backend: {args.inference_backend}. Available backends: gyro, stub")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Setup model with appropriate parameters based on backend
    try:
        if args.inference_backend == "gyro":
            infer_next_token = gyro_setup_model(encoding, args.config)
        elif args.inference_backend == "stub":
            infer_next_token = stub_setup_model()
        else:
            raise ValueError(f"Unsupported backend: {args.inference_backend}")
    except Exception as e:
        print(f"FATAL: Engine initialization failed: {e}")
        print("This is typically caused by:")
        print("  - Missing or corrupted model files")
        print("  - Invalid configuration")
        print("Please check your configuration and model files.")
        exit(1)

    uvicorn.run(create_api_server(infer_next_token, encoding), port=args.port)
