import argparse
import yaml

from engine.train import train_process, set_seeds

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the YAML configuration file', required=True)

    ### Network structure
    parser.add_argument('--model_name', type=str, help='the model name')
    parser.add_argument('--is_aug', action='store_true', help='Using augmentation or not')
    parser.add_argument('--resolution', type=int, default=64, required=False, help='model config')
    parser.add_argument('--alpha', type=float, default=0.6, required=False, help='model config')
    # parser.add_argument('--alpha', type=float, default=0.75, required=False, help='model config')

    ### Dataset setting
    parser.add_argument('--path_dir', type=str, help='dataset path_dir')
    parser.add_argument('--batch_size', type=int, help='batch_size')
    parser.add_argument('--transform', type=str, help='basic augmentation')
    parser.add_argument('--augment_ratio', type=int, help='augment_ratio')
    parser.add_argument('--fold', type=int, help='training fold')

    ### Training setting
    parser.add_argument('--epoch', type=int, help='epochs')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--model_dir', type=str, help='saved model dir')
    parser.add_argument('--result_dir', type=str, help='result_dir')

    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seeds(args.seed)
    # Load the YAML configuration file
    yaml_config = load_yaml_config(args.config)

    # Override YAML values with command-line arguments if provided
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config' and arg != 'seed':
            section = None
            if arg in ['model_name', 'is_aug']:
                section = 'network_structure'
            elif arg in ['path_dir', 'fold', 'batch_size', 'transform', 'augment_ratio']:
                section = 'dataset'
            elif arg in ['learning_rate', 'epochs', 'use_gpu', 'model_dir', 'result_dir']:
                section = 'training_setting'
            
            if section:
                if section not in yaml_config:
                    yaml_config[section] = {}
                yaml_config[section][arg] = getattr(args, arg)

        if arg in ['resolution', 'alpha']:
            yaml_config['network_structure']['model_config'][arg] = getattr(args, arg)

    # Print the final configuration
    print("Final Configuration:", yaml_config)
    return yaml_config

if __name__ == '__main__':
    config = main()
    train_process(config)
