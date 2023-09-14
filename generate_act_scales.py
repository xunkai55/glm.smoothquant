import torch
import os

from transformers import AutoModel, AutoTokenizer
import argparse

from smoothquant.calibration import get_act_scales

def build_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model-path', type=str, required=True)
    parser.add_argument('-o', '--output-path', type=str, required=True,
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='location of the calibration dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_path)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        raise FileNotFoundError

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()
