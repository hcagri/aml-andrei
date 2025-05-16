import time
import logging
from src.util import set_seed, extract_param
from src.dataloader import AMLData
from src.training import train_gnn

import random
from datetime import datetime
import os
import sys
import yaml
import argparse
import torch
import wandb
from types import SimpleNamespace
from pprint import pprint

def create_parser():
    parser = argparse.ArgumentParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #rint(f"Current directory: {current_dir}")
    default_data_path = os.path.join(current_dir, "data")
    default_output_dir = os.path.join(current_dir, "results")
    # test to check if i set the github account back to normal
    parser.add_argument("--data_path", default=default_data_path, type=str, help="Path to the data directory")
    parser.add_argument("--output_dir", default=default_output_dir, type=str, help="Path to the output directory")
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    
    #Model parameters
    parser.add_argument("--batch_size", default=4096, type=int, help="Select the batch size for GNN training")
    parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', default=[100,100], help='Pass the number of neighors to be sampled in each hop (descending).')

    #Misc
    parser.add_argument("--device", default="cuda:0", type=str, help="Select a GPU", required=False)
    parser.add_argument("--seed", default=1, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, pna]", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")

    return parser

def logger_setup(log_dir: str):
    """Setup logging to file and stdout"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "logs.log")),  # Log to run-specific log file
            logging.StreamHandler(sys.stdout)                        # Log also to stdout
        ]
    )

def setup_config(args):
    """Setup configuration and logging directories, consolidating all args into config
    
    Args:
        args: Command line arguments
        
    Returns:
        SimpleNamespace: Configuration object with attribute-style access
    """
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.data}_{timestamp}")
    log_dir = os.path.join(run_dir, "logs")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger_setup(log_dir)   
    
    # Create config dictionary with all args included
    config_dict = create_config_dict(args, run_dir, log_dir, checkpoint_dir)

    # Print the config 
    print("----- CONFIG -----")
    for key, value in config_dict.items():
        pprint(f"{key}: {value}")
    print(2*"------------------")

    # Save config to file
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logging.info(f"Configuration saved to {config_path}")
    # 
    if not args.testing:
        # Initialize wandb
        wandb.init(
            project=f"AML_{args.data}",
            name=f"{args.data}_{args.model}_{timestamp}",
            config=config_dict,
            dir=run_dir,
            reinit=True
        )

    # Convert dictionary to SimpleNamespace for attribute-style access
    config = SimpleNamespace(**config_dict)
    
    return config

def create_config_dict(args, run_dir, log_dir, checkpoint_dir):
    # Create config dictionary with all args included
    config_dict = {
        # Training configuration
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "model": args.model,
        "data": args.data,
        "num_neighs": args.num_neighs,
        
        
        # Directories
        "run_dir": run_dir,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": args.output_dir,
        "data_path": args.data_path,
        
        # Other args
        "seed": args.seed,
        "device": torch.device(args.device if torch.cuda.is_available() else "cpu"),
        "emlps": args.emlps,
        "save_model": args.save_model if hasattr(args, 'save_model') else False
    }

    if args.model == "interleaved" or args.model == "fusion":
        arch = extract_param("architecture", args)
        arch_params = []
        for model in arch:
            temp_dict = {}
            
            if model == "gin" or model == "pna":
                temp_dict["model"] = model
                temp_dict["lr"] = extract_param("lr", args, model)
                temp_dict["n_hidden"] = extract_param("n_hidden", args, model)
                temp_dict["n_gnn_layers"] = extract_param("n_gnn_layers", args, model)
                temp_dict["loss"] = "ce"
                temp_dict["w_ce1"] = extract_param("w_ce1", args,model)
                temp_dict["w_ce2"] = extract_param("w_ce2", args,model)
                temp_dict["dropout"] = extract_param("dropout", args,model)
                temp_dict["final_dropout"] = extract_param("final_dropout", args,model)
            elif model == "transformer":
                temp_dict["model"] = model
                temp_dict["no_heads"] = extract_param("no_heads", args, model)
                temp_dict["n_hidden"] = extract_param("n_hidden", args, model)
                temp_dict["n_layers"] = extract_param("n_layers", args, model)
                temp_dict["activation"] = extract_param("activation", args, model)
                temp_dict["dropout"] = extract_param("dropout", args, model)
            elif model == "fmlp":
                temp_dict["model"] = model
                temp_dict["n_hidden"] = extract_param("n_hidden", args, model)
                temp_dict["dropout"] = extract_param("dropout", args, model)
                temp_dict["activation"] = extract_param("activation", args, model)

            elif model == "gmu":
                raise NotImplementedError("GMU fusion not implemented")    
            arch_params.append(temp_dict)
        
        config_dict["arch"] = arch_params

        config_dict["batch_accum"] = extract_param("batch_accum", args)
        config_dict["clip_grad"] = extract_param("clip_grad", args)

        config_dict["n_hidden"] = extract_param("n_hidden", args)
        config_dict["final_dropout"] = extract_param("final_dropout", args)
        config_dict["lr"] = extract_param("lr", args)
        config_dict["loss"] = "ce"
        config_dict["w_ce1"] = extract_param("w_ce1", args)
        config_dict["w_ce2"] = extract_param("w_ce2", args)

    else:
        config_dict["batch_accum"] = extract_param("batch_accum", args)
        config_dict["clip_grad"] = extract_param("clip_grad", args)
        config_dict["lr"] = extract_param("lr", args)
        config_dict["n_hidden"] = extract_param("n_hidden", args)
        config_dict["n_gnn_layers"] = extract_param("n_gnn_layers", args)
        config_dict["loss"] = "ce"
        config_dict["w_ce1"] = extract_param("w_ce1", args)
        config_dict["w_ce2"] = extract_param("w_ce2", args)
        config_dict["dropout"] = extract_param("dropout", args)
        config_dict["final_dropout"] = extract_param("final_dropout", args)

    
    # Add any other args that weren't explicitly handled
    for key, value in vars(args).items():
        if key not in config_dict:
            config_dict[key] = value

    return config_dict


def main():
    parser = create_parser()
    args = parser.parse_args()
    args.num_neighs = [int(t) for t in args.num_neighs]

    # Setup configuration 
    config = setup_config(args)

    # Set seed
    if args.seed == 1:
        args.seed = random.randint(2, 256000)
    set_seed(args.seed)

    # Get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    dataset = AMLData(config)  # Use config instead of args
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = dataset.get_data()
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    # Training (only passing config, not args)
    logging.info(f"Running Training")
    train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, config)


if __name__ == "__main__":
    main()
