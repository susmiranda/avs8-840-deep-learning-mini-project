import os

import torch

'''
Script to change an SSformer saved model to match KWT.
'''

def load_model(model_path:str) -> dict:
    return torch.load(model_path, map_location=device)

def change_keys(model:dict, lightning:bool) -> dict:
    # Remove all keys that are not for KWT
    for item in [key for key in model['state_dict'].keys() if 'model.encoder' not in key]:
        model['state_dict'].pop(item)

    # Remove 'model.encoder' from the start of all KWT keys
    if lightning:
        for item in model['state_dict'].copy().keys():
            model['state_dict'][item.replace('model.encoder.', 'model.')] = model['state_dict'].pop(item)
    else:
        for item in model['state_dict'].copy().keys():
            model['state_dict'][item.replace('model.encoder.', '')] = model['state_dict'].pop(item)
    return model

def save_model(model:dict, out_dir:str, filename:str):
    print(f'Saving model to: {os.path.join(out_dir, filename)}')
    torch.save(model, os.path.join(out_dir, filename))

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--model_path', type=str, required=True, help='Path to the ssformer model checkpoint')
    ap.add_argument('--out_dir', type=str, required=False, help='Directory to save the new checkpoint.')
    ap.add_argument('--filename', type=str, required=False, help='Name of the new checkpoint.')
    ap.add_argument('--lightning', action='store_true', default=False, help='Choose for lightning compatible model.')
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model(args.model_path)
    model = change_keys(model, args.lightning)

    out_dir = args.out_dir if args.out_dir else os.path.dirname(args.model_path)
    filename = args.filename if args.filename else f"{os.path.basename(args.model_path).split('.')[0]}_KWT.ckpt"
    save_model(model, out_dir, filename)