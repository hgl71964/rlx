import os
import torch

# the path to save the onnx model
ONNX_PATH = './data/models/'

MODELS = [
    # vision
    # "mobilenet_v2",
    # "inception_v3",
    # "resnext50_32x4d",
    #"resnext101_32x8d",
    #"resnet18",
    #"resnet34",
    # "resnet50",
    # "resnet101",
    #"resnet152",

    # nlp
    "bert-base-uncased",
    # "bert-large-uncased",
    "bert-base-cased",
    # "bert-large-cased",
    "gpt2",
]


def main():

    print(torch.hub.list("pytorch/vision"))
    # print(torch.hub.list("huggingface/pytorch-transformers"))
    # print(torch.hub.list("pytorch/fairseq"))
    # print(torch.hub.list("pytorch/text"))
    for model in MODELS:
        full_path = os.path.join(ONNX_PATH, model)
        if os.path.exists(full_path):
            print(f"{full_path} exist!")
            continue
        else:
            os.makedirs(full_path)

        if model[:6] == "resnet":
            # load pretrained resnet50 and create a random input
            torch_model = torch.hub.load('pytorch/vision:v0.6.0',
                                         model,
                                         pretrained=True,
                                         verbose=False)
            torch_model = torch_model.cuda().eval()
            torch_data = torch.randn([1, 3, 224, 224]).cuda()

            # export the pytorch model to onnx model 'resnet50.onnx'
            torch.onnx.export(
                model=torch_model,
                args=torch_data,
                f=os.path.join(full_path, model + ".onnx"),
                input_names=['data'],
                output_names=['output'],
                dynamic_axes={
                    'data': {
                        0: 'batch_size'
                    },
                    'output': {
                        0: 'batch_size'
                    }
                },
            )

        elif model == "resnext50_32x4d":
            torch_model = torch.hub.load('pytorch/vision:v0.6.0',
                                         model,
                                         pretrained=True,
                                         verbose=False)
            torch_model = torch_model.cuda().eval()
            torch_data = torch.randn([1, 3, 224, 224]).cuda()
            torch.onnx.export(
                model=torch_model,
                args=torch_data,
                f=os.path.join(full_path, model + ".onnx"),
                input_names=['data'],
                output_names=['output'],
            )

        elif model == "mobilenet_v2":
            torch_model = torch.hub.load('pytorch/vision:v0.6.0',
                                         'mobilenet_v2',
                                         pretrained=True)
            torch_model = torch_model.cuda().eval()
            torch_data = torch.randn([1, 3, 224, 224]).cuda()
            torch.onnx.export(
                model=torch_model,
                args=torch_data,
                f=os.path.join(full_path, model + ".onnx"),
                input_names=['data'],
                output_names=['output'],
                # dynamic_axes={'data': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            )

        elif model == "inception_v3":
            torch_model = torch.hub.load('pytorch/vision:v0.6.0',
                                         'inception_v3',
                                         pretrained=True)
            torch_model = torch_model.cuda().eval()
            torch_data = torch.randn([1, 3, 224, 224]).cuda()

            # export the pytorch model to onnx model 'resnet50.onnx'
            torch.onnx.export(
                model=torch_model,
                args=torch_data,
                f=os.path.join(full_path, model + ".onnx"),
                input_names=['data'],
                output_names=['output'],
                # dynamic_axes={'data': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            )
        elif model[:4] == "bert" or model == "gpt2":
            batch_size = 1
            seq_length = 128
            repo_name = 'huggingface/pytorch-transformers'
            torch_model = torch.hub.load(repo_name, 'model', model)
            torch_model = torch_model.cuda().eval()

            # input
            tokens_tensor = torch.zeros((batch_size, seq_length),
                                        dtype=torch.long,
                                        device='cuda')
            segments_tensors = torch.zeros((batch_size, seq_length),
                                           dtype=torch.long,
                                           device='cuda')
            args = (tokens_tensor, )
            kwargs = {'token_type_ids': segments_tensors}

            # NOTE: consider convert to other precision; f16, f32, f64
            # def convert_f32(arg):
            #     if arg.dtype == torch.float32:
            #         return arg.to(dtype=BenchModel.dtype)
            #     return arg
            # args = [convert_f32(arg.cuda()) for arg in args]
            # kwargs = {k: convert_f32(v.cuda()) for k, v in kwargs.items()}

            #dummy_input = args, kwargs
            dummy_input = tokens_tensor
            torch.onnx.export(
                torch_model,
                args=dummy_input,
                f=os.path.join(full_path, model + ".onnx"),
                input_names=["input_ids"],
                output_names=["output_hidden_states", "output_attentions"],
                # dynamic_axes={
                #     "token": {
                #         0: "batch_size",
                #         1: "sequence_length"
                #     },
                #     "output_hidden_states": {
                #         0: "batch_size",
                #         1: "sequence_length"
                #     },
                #     "output_attentions": {
                #         0: "batch_size",
                #         1: "sequence_length"
                #     },
                # },
                # opset_version=11,  # Choose an appropriate ONNX opset version
            )

        else:
            raise RuntimeError(f"Unsupported model: {model}")

        # print('{}: {:.1f} MiB'.format(full_path,
        #                               os.path.getsize(full_path) / (2**20)))


if __name__ == '__main__':
    main()
