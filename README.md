# Deploying Machine Learning Models with PyTorch, gRPC and asyncio

Today we're going to see how to deploy a machine-learning model behind gRPC service running via asyncio. gRPC promises to be faster, more scalable and more optimized than HTTP. We will use PyTorch to create an image classifier and perform inference using gRPC calls.

This article is also hosted on [GitHub](https://github.com/FrancescoSaverioZuppichini/deploy-models-with-grpc-pytorch-asyncio)

## What's gRPC

What's [gRPC](https://grpc.io/)? GRPC is a Remote Procedure Call (RPC) framework that runs on any device. It's developed and maintained mainly by Google and it's widely used in the industry. It allows two machines to communicate, similar to HTTP but with better syntax and performance. It's used to define microservices that may use different programming languages.

It works by defining the fields of the messages the client and server will exchange and the signature of the function we will expose, with a special syntax in a `.proto` file, then gRPC generates both client and server code and you can call the function directly from the client.

gRPC services send and receive data as Protocol Buffer (Protobuf) messages, they can be better compressed than human-readable format (like JSON or XML), thus the better performance.

## Getting Started

Let's start by setup our environment using virtual env

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
- `server.py` holds the server code responsible of receiving the inference request and sending a reply
- `inference.py` holds the actual model and inference logic
- `inference.proto` holds the protocol buffer messages definition

Let's start by coding our model inside `inference.py`


## Inference

We will use `resnet34` from `torchvision`. First thing, we define our preprocessing transformation

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

Sweet, now the model

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

The model will output `262`, which is the right class for our `cat`. Our `inference` function takes a list of `Pil` images and creates a batch, then it collects the right classes and converts them to a list of class ids.

Nice, we have our model setup.

## Server

The next step is to create the actual gRPC server. First, we describe the message and the service in the `.proto` file. 

A list of all types of messages can be found [here](https://learn.microsoft.com/en-us/dotnet/architecture/grpc-for-wcf-developers/protocol-buffers) and the official python tutorial for gRPC [here](https://grpc.io/docs/languages/python/basics/)

### Proto

We will start by defining our `InferenceServer` service

```proto
// inference.proto

syntax = "proto3";

// The inference service definition.
service InferenceServer {
  // Sends a inference reply
  rpc inference (InferenceRequest) returns (InferenceReply) {}
}

```

This tells gRPC we have an `InferenceServer` service with an `inference` function, notice that we need to specify the type of the messages: `InferenceRequest` and `InferenceReply`

```proto
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

Our request will send a list of bytes (images), the `repeated` keyword is used to define lists, and we will send back a list of predictions

### Build the server and client

Now, we need to generate the client and server code using `grpcio-tools` (we install it at the beginning). 

```bash
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

- `inference_pb2_grpc` contains our gRPC's server definition
- `inference_pb2` contains our gRPC's messages definition
- `inference_pb2` contains our gRPC's messages types definition

We now have to code our service, 

```python
# server.py
# we will use asyncio to run our service
import asyncio 
...
# from the generated grpc server definition, import the required stuff
from inference_pb2_grpc import InferenceServer, add_InferenceServerServicer_to_server
# import the requests and reply types
from inference_pb2 import InferenceRequest, InferenceReply
...
```

To create the gRPC server we need to import `InferenceServer` and `add_InferenceServerServicer_to_server` from the generated `inference_pb2_grpc`. Our logic will go inside a subclass of `InferenceServer` in the `inference` function, the one we defined in the `.proto` file.

```python
# server.py
class InferenceService(InferenceServer):
    def open_image(self, image: bytes) -> Image.Image:
        image = Image.open(BytesIO(image))
        return image

    async def inference(self, request: InferenceRequest, context) -> InferenceReply:
        logging.info(f"[🦾] Received request")
        start = perf_counter()
        images = list(map(self.open_image, request.image))
        preds = inference(images)
        logging.info(f"[✅] Done in {(perf_counter() - start) * 1000:.2f}ms")
        return InferenceReply(pred=preds)
```

Notice we subclass `InferenceServer`, we add our logic inside `inference` and we label it as an `async` function, this is because we will lunch our service using [asyncio](https://docs.python.org/3/library/asyncio.html). 

We now need to tell gRPC how to start our service.

```python
# server.py
...
from inference_pb2_grpc import InferenceServer, add_InferenceServerServicer_to_server
import logging

logging.basicConfig(level=logging.INFO)

async def serve():
    server = grpc.aio.server()
    add_InferenceServerServicer_to_server(InferenceService(), server)
    # using ip v6
    adddress = "[::]:50052"
    server.add_insecure_port(adddress)
    logging.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()
```

Line by line, we create a grpc asyncio server using `grpc.aio.server()`, we add our service by passing it to `add_InferenceServerServicer_to_server` then we listed on a custom port using ipv6 by calling the `.add_insecure_port` method and finally we await the `.start` server method

Finally, 

```python
# server.py
if __name__ == "__main__":
    asyncio.run(serve())
```

If you know run the file

```bash
python src/server.py
```

You'll see

```
INFO:root:[📡] Starting server on [::]:50052
```

The full server looks like

```python
import asyncio
from time import perf_counter

import grpc
from PIL import Image
from io import BytesIO
from inference import inference
import logging
from inference_pb2_grpc import InferenceServer, add_InferenceServerServicer_to_server
from inference_pb2 import InferenceRequest, InferenceReply

logging.basicConfig(level=logging.INFO)


class InferenceService(InferenceServer):
    def open_image(self, image: bytes) -> Image.Image:
        image = Image.open(BytesIO(image))
        return image

    async def inference(self, request: InferenceRequest, context) -> InferenceReply:
        logging.info(f"[🦾] Received request")
        start = perf_counter()
        images = list(map(self.open_image, request.image))
        preds = inference(images)
        logging.info(f"[✅] Done in {(perf_counter() - start) * 1000:.2f}ms")
        return InferenceReply(pred=preds)


async def serve():
    server = grpc.aio.server()
    add_InferenceServerServicer_to_server(InferenceService(), server)
    # using ip v6
    adddress = "[::]:50052"
    server.add_insecure_port(adddress)
    logging.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
```


Sweet 🎉! We have our gRPC running with asyncio. We now need to define our **client**.

## Client

Creating a client is straightforward, similar to before we need the definitions that were generated in the previous step.

```python
# client.py

import asyncio

import grpc

from inference_pb2 import InferenceRequest, InferenceReply
from inference_pb2_grpc import InferenceServerStub
```

`InferenceServerStub` is the gRPC communication point. Let's create our `async` function to send `InferenceRequest` and collect `InferenceReply`

```python
...
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    async with grpc.aio.insecure_channel("[::]:50052 ") as channel:
        stub = InferenceServerStub(channel)
        start = perf_counter()

        res: InferenceReply = await stub.inference(
            InferenceRequest(image=[image_bytes])
        )
        logging.info(
            f"[✅] pred = {pformat(res.pred)} in {(perf_counter() - start) * 1000:.2f}ms"
        )
```

We define our channel using `grpc.aio.insecure_channel` context manager, we create an instance of `InferenceServerStub` and we `await` the `.inference` method. The `.inference` method takes `InferenceRequest` instance containing our images in `bytes`. We receive back an `InferenceReply` instance and we print the predictions.

To get the bytes from an image, we can use `Pillow` and `BytesIO`

```python
from io import BytesIO
from PIL import Image

# client.py

image = Image.open("./examples/cat.jpg")
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_bytes = buffered.getvalue()
```

The full client code looks like

```python
import asyncio
from io import BytesIO

import grpc
from PIL import Image

from inference_pb2 import InferenceRequest, InferenceReply
from inference_pb2_grpc import InferenceServerStub
import logging
from pprint import pformat
from time import perf_counter

image = Image.open("./examples/cat.jpg")
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_bytes = buffered.getvalue()

logging.basicConfig(level=logging.INFO)


async def main():
    async with grpc.aio.insecure_channel("[::]:50052 ") as channel:
        stub = InferenceServerStub(channel)
        start = perf_counter()

        res: InferenceReply = await stub.inference(
            InferenceRequest(image=[image_bytes])
        )
        logging.info(
            f"[✅] pred = {pformat(res.pred)} in {(perf_counter() - start) * 1000:.2f}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
```

let's run it!

```bash
python src/client.py
```

It results in the following output in the client

```
// client
INFO:root:[✅] pred = [282] in 86.39ms
```

and on the server

```
// server
INFO:root:[🦾] Received request
INFO:root:[✅] Done in 84.03ms
```

Nice!!! We can also pass multiple images, 

```python
# client.py
...
        res: InferenceReply = await stub.inference(
                    InferenceRequest(image=[image_bytes, image_bytes, image_bytes])
                )
```

We just copied and pasted `[image_bytes, image_bytes, image_bytes]` to send 3 images

If we run it,

```bash
python src/client.py
```

We get

```
INFO:root:[✅] pred = [282, 282, 282] in 208.39ms
```

yes, 3 predictions on the same gRPC call! 🚀🚀🚀

## Conclusion

Today we have seen how to deploy a machine learning model using PyTorch, gRPC and asyncio. A scalable, effective and performant to make your model accessible. There are many gRPC features we didn't touch like [streaming](https://grpc.io/docs/what-is-grpc/core-concepts/#server-streaming-rpc). 

I hope it helps!

See you in the next one,

Francesco