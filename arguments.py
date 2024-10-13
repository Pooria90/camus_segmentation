import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="This is our segmentation script.")

    '''
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    '''

    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="The path to load the model from."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result-seg/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
        ),
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size for the training dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=300)

    parser.add_argument(
        "--valid_batch_size", type=int, default=8, help="Batch size for the validation dataloader."
    )

    parser.add_argument(
        "--model", type=str, default="unet_small", help="Segmentation model type to be trained/fine-tuned."
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    
    parser.add_argument(
        "--train_frames",
        type=str,
        default='./data-echo-pat/train-frames-efn/',
        help=(
            "Train frames directory."
        ),
    )

    parser.add_argument(
        "--aug_train_frames",
        type=str,
        default=None,
        help=(
            "Directory to the augmented training frames."
        ),
    )

    parser.add_argument(
        "--train_masks",
        type=str,
        default='./data-echo-pat/train-masks-efn/',
        help=(
            "Train masks directory."
        ),
    )

    parser.add_argument(
        "--aug_train_masks",
        type=str,
        default=None,
        help=(
            "Directory to the augmented training masks."
        ),
    )

    parser.add_argument(
        "--aug_prop",
        type=float,
        default=1.0,
        help=(
            "The proportion taken from augmented training data."
        ),
    )

    parser.add_argument(
        "--valid_frames",
        type=str,
        default='./data-echo-pat/valid-frames-efn/',
        help=(
            "Valid frames directory."
        ),
    )

    parser.add_argument(
        "--valid_masks",
        type=str,
        default='./data-echo-pat/valid-masks-efn/',
        help=(
            "Valid masks directory."
        ),
    )

    '''
    parser.add_argument(
        "--save_image_epochs",
        type=int,
        default=15,
        help=(
            "Dummy save image epochs!"
        ),
    )
    '''

    parser.add_argument(
        "--val_interval",
        type=int,
        default=5,
        help=(
            "validation period!"
        ),
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help=(
            "Number of semantic classes!"
        ),
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help=(
            "GPU number."
        )
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args