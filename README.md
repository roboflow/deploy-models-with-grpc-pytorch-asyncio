# Deploying Machine Learning Models with gRPC

Today we're going to see how to deploy a machine learning model behind gRPC. We will use PyTorch to create an image classifier and performing inference using gRPC calls.

## What's gRPC

What's [gRPC](https://grpc.io/)? GRPC is a Remote Produre Call (RPC) framework that runs on any device. It's develop and mainted mainly by Google and it's widely used in the industry. It allows two machine to communicate, similar to HTTP but with better syntax and performance. It's used to define microservices that may use different programming languages.

It works by defining the fields of the messages client and server will exchange and the signature of the function we will expose, with a special syntax in a `.proto` file, then gRPC generates both client and server code and you can call the function directly from the client.

gRPC services send and receive data as Protocol Buffer (Protobuf) messages, they can be better compress than human readable format (like JSON or XML), thus the better performance.

## Getting Started

Let's start by setup our enviroment by creating a virtual env

**Tested with python 3.9**

```
python -m venv .venv
```

Then, let's install all the required packages, `grpcio`, `grpcio-tools`, `torch`, `torchvision` and `Pillow`

```
pip install grpcio grpcio-tools torch torchvision Pillow==9.3.0
```

All set!

We will work on 4 files,

```
.
└── src
    ├── client.py
    ├── inference.proto
    ├── inference.py
    └── server.py
```

- `client.py` holds the client code we will use to send inference requests
- `server.py` holds the server code responsable of receiving the inference request and sending a reply
- `inference.py` holds the actual model and inference logic
- `inference.proto` holds the protocol buffer messages definition

Let's start by coding our model inside `inference.py`


## Inference

We will use `resnet34` from `torchvision`, let's start by defining our preprocessing transformation

```python
# inference.py
import torchvision.transforms as T

preprocess = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


if __name__ == "__main__":
    from PIL import Image
    image = Image.open('./examples/cat.jpg')
    tensor = preprocess(image)
    print(tensor.shape)
```

Sweet, let's add the model

```python
# inference.py
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet34_Weights, resnet34

preprocess = ...
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).eval()


@torch.no_grad()
def inference(images: List[Image.Image]) -> List[int]:
    batch = torch.stack([preprocess(image) for image in images])
    logits = model(batch)
    preds = logits.argmax(dim=1).tolist()
    return preds


if __name__ == "__main__":
    image = Image.open("./examples/cat.jpg")
    print(inference([image]))

```

The model will output `262`, that is the right class for our `cat`. Our `inference` function takes a list of `Pil` images and create a batch, then it collects the right classes and convert them to a list of class ids.

Nice, we have our model setup.

## Server

Next step is to create the actual gRPC server, the first thing to do is to describe the message and the service to gRPC in the `.proto` file. 

A list of all types for the messages can be find [here](https://learn.microsoft.com/en-us/dotnet/architecture/grpc-for-wcf-developers/protocol-buffers) and the official python tutorial for gRPC [here](https://grpc.io/docs/languages/python/basics/)

### Proto

We will start by defining our `InferenceServer` service

```
// inference.proto

syntax = "proto3";

// The inference service definition.
service InferenceServer {
  // Sends a inference reply
  rpc inference (InferenceRequest) returns (InferenceReply) {}
}

```

This tells grpc we have a `InferenceServer` service with a `inference` function, notice that we need to specify the type of the messages: `InferenceRequest` and `InferenceReply`

```
// inference.proto
...
// The request message containing the images.
message InferenceRequest {
    repeated bytes image = 1;
}

// The response message containing the classes ids
message InferenceReply {
    repeated uint32 pred = 1;
}
```

Our request will send a list of bytes (images), the `repeated` keyword is used to defined lists, and we will send back a list of predictions

### Build the server and client

Now, we need to generate the client and server code using `grpcio-tools` (we install it at the beginning). 

```
cd src && python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. inference.proto 
```

This will generate the following files

```
└── src
    ├── inference_pb2_grpc.py
    ├── inference_pb2.py
    ├── inference_pb2.pyi
    ...
```

- `inference_pb2_grpc` contains our grpc server definition
- `inference_pb2` contains our grpc messages definition
- `inference_pb2` contains our grpc messages types definition
