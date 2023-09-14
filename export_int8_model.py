import torch
import argparse
import os

from pathlib import Path

from transformers import AutoTokenizer, AutoModel

from smoothquant.smooth import smooth_lm

from smoothquant.calibration import get_static_decoder_layer_scales


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model-path", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-a", "--alpha", type=float, default=0.7)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='location of the calibration dataset')
    parser.add_argument('--export-FT', default=False, action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True)
    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, args.alpha)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        raise FileNotFoundError

    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       args.dataset_path,
                                                                       num_samples=args.num_samples,
                                                                       seq_len=args.seq_len)
    output_path = Path(args.output_path) / 'auto-model'
    if args.export_FT:
        model.save_pretrained(output_path)
        print(f"Saved smoothed model at {output_path}")

        output_path = Path(args.output_path) / (Path(args.model_path).name + "-sq-scales.pt")
        torch.save(raw_scales, output_path)
        print(f"Saved scaling factors at {output_path}")
    else:
        from smoothquant.opt import Int8OPTForCausalLM

        int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
        int8_model.save_pretrained(output_path)
        print(f"Saved int8 model at {output_path}")
