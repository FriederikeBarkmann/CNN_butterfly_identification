import os
import argparse
import time, datetime
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter, OrderedDict
import torch
import torchvision
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision.models import resnet152
from torch.utils.data import Dataset, Subset, DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed


# Parse command line arguments
parser = argparse.ArgumentParser(description='Butterflies with DDP',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default="resnet152",
                    help='Pretrained model to be finetuned')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='starting learning rate')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs that meet target before stopping')
parser.add_argument('--data-dir', type=str, default="/scratch",
                    help='directory with butterfly dataset and indices')
parser.add_argument('--results-dir', type=str, default="/scratch",
                    help='directory to store results and checkpoints')
args = parser.parse_args()


def worker(args):
    # Oversample minority classes, or weights to loss function, or apply none (just changed)
    treat_imbalance = "oversampling" # oversampling, weighted_loss, none
    # Keep reproducible (Accelerate sets all seeds)
    set_seed(42)

    # Folders for results and checkpoints
    results_folder = os.path.join(args.results_dir, "butterflies", "results",
        f"{args.model}",
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    #checkpoints_folder = os.path.join(results_folder, "checkpoints")
    #if accelerator.is_main_process:
    #    os.makedirs(checkpoints_folder)

    # Use ProjectConfiguration
    project_config= ProjectConfiguration(project_dir=results_folder,
                                         automatic_checkpoint_naming=True)

    # Initialize HF accelerator
    accelerator = Accelerator(project_config=project_config)

    # GPU
    device = accelerator.device
    # Check number of processes
    if accelerator.is_main_process:
        print(f"Number of processes: {accelerator.num_processes}")


    # ======== Data Preparation ========


    # Transformations of training data to avoid overfitting
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size = 224, scale = (0.5, 1), ratio = (0.8, 1.2)), #randomly crop image between to up to half the image, randomly change the aspect ratio, and crop to 224 x 224 px
        v2.RandomHorizontalFlip(p = 0.3),
        v2.RandomVerticalFlip(p = 0.3),
        v2.RandomPerspective(distortion_scale = 0.2, p = 0.4), #Randomly distort the perspective of the image
        v2.RandomRotation(degrees = 50, expand = False), #Random Rotation from -50 to +50 dgrees
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #normalize based on ImageNet
    ])

    # Transformations of validation data (only resizes and crops images and normalizes)
    val_transform = v2.Compose([
        v2.Resize(size = 224),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Connect dataset
    data_dir = os.path.join(args.data_dir, "Schmetterlinge_sym")
    dataset = ImageFolder(root = data_dir)
    # Load both training and validation but not test subset (indices) of the data
    idx_train_val = np.load(args.data_dir + "/data_prep/train_idx.npy")


    # Class names
    classes = dataset.classes
    #accelerator.print(f"Classes: {classes}")
    # Number of classes
    n_classes = len(classes)  # -> Used for final classification layer
    #accelerator.print(f"Number of classes: {n_classes}")
    # Number of examples per class -> used for train-test-split
    targets = [dataset.targets[i] for i in idx_train_val]
    #accelerator.print(f"Targets: {targets[:15]}...")
    #accelerator.print("Examples per class: {}".format(Counter(targets)))


    # Split indices into training and validation sets
    train_idx, val_idx = train_test_split(
        idx_train_val,
        test_size = 0.2,
        shuffle = True,
        stratify = targets  # Ensure similar number of examples per class
    )


    # Set number of muliprocessing workers for the data loaders
    num_workers = int(os.environ['OMP_NUM_THREADS'])


    # Load training data
    train_data = ImageFolder(root = data_dir, transform = train_transform)
    train_data = Subset(train_data, train_idx)

    # Oversampling of minority classes
    if treat_imbalance == "oversampling":
        sample_startTime = time.time()
        accelerator.print("Oversampling training data.")
        # Count the number of images per class (replace with targets?)
        labels = [label for image, label in train_data]
        counts = Counter(labels)
        accelerator.print(counts)
        # Get the weight for each class (inverse value of the number of images in each class)
        class_weights_dict = dict(zip(counts.keys(),
                    [1/weights for weights in list(counts.values())]))
        # Assign the weights to each sample in the unbalanced dataset
        sample_weights = [class_weights_dict.get(i) for i in labels]
        # Total number of samples to be drawn (same as length of training dataset)
        # num_samples = len(train_dl.dataset)
        # Oversample minority classes with a random sampler
        train_sampler = torch.utils.data.WeightedRandomSampler(
                            weights = sample_weights,
                            num_samples = len(train_idx),
                            replacement = True)
        sample_endTime = time.time()
        accelerator.print("Done. Time used to sample training data: {:.2f}s".format(sample_endTime-sample_startTime))
        # Data loader
        train_dl = DataLoader(train_data,
                batch_size = args.batch_size,
                sampler = train_sampler,
                drop_last = True,
                num_workers = num_workers)

    else:
        # Data loader without weighted sampling
        train_dl = DataLoader(train_data,
                batch_size = args.batch_size,
                drop_last = True,
                num_workers = num_workers)

    # Load valiation data with fewer trafos (and without reweighting)
    val_data = ImageFolder(root = data_dir,
            transform = val_transform)
    val_data = Subset(val_data, val_idx)
    val_dl = DataLoader(val_data,
            batch_size = args.batch_size,
            drop_last = True,
            num_workers = num_workers,
            pin_memory = True) # pin memory?


    # ======== Training ========

    startTime = time.time()
    if accelerator.is_main_process:
        print("Start training.")

    # Set the pre-trained model
    # exec(f"model = {args.model}(weights='DEFAULT')")  # exec troubles
    model = eval(f"{args.model}(weights='DEFAULT')")

    # Adapt the last layer to classes of the dataset for finetuning
    if ("resne" in args.model or "regne" in args.model):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, n_classes))
    elif (args.model.startswith("densenet")):
        if (args.model.endswith("201")):
            model.classifier=nn.Linear(1920, n_classes)
        elif (args.model.endswith("169")):
            model.classifier=nn.Linear(1664, n_classes)
        elif (args.model.endswith("161")):
            model.classifier=nn.Linear(2208, n_classes)
        elif (args.model.endswith("121")):
            model.classifier=nn.Linear(1024, n_classes)
    elif (args.model.startswith("vgg")):
        num_ftrs = 512*7*7
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
            nn.LogSoftmax(dim=1))
    elif (args.model.startswith("efficientnet")):
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(1280, n_classes),)
    elif (args.model.startswith("vit")):
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if (args.model.startswith("vit_b")):
            hidden_dim=768
        elif (args.model.startswith("vit_l")):
            hidden_dim=1024
        elif (args.model.startswith("vit_h")):
            model = vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')
            hidden_dim=1280
        heads_layers["head"] = nn.Linear(hidden_dim, n_classes)
        model.heads = nn.Sequential(heads_layers)
    elif (args.model.startswith("swin")):
        if (args.model.endswith("_t")):
            embed_dim=96
        elif (args.model.endswith("_s")):
            embed_dim=96
        elif (args.model.endswith("_b")):
            embed_dim=128
        num_features = embed_dim * 2 ** 3
        model.head = nn.Linear(num_features, n_classes)
    elif (args.model.startswith("maxvit")):
        block_channels=[64, 128, 256, 512]
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], n_classes, bias=False)
        )


    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr = args.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = opt, # reduces learning when loss stops decreasing
                mode = "min",
                factor = 0.5, # factor by which the learning rate is increased
                patience = 2) # number of epochs without improvement until learning rate decreases
    accelerator.register_for_checkpointing(scheduler)


    # Cross-Entropy Loss
    if treat_imbalance == "weighted_loss":
        # Estimation of class weights using original train+val dataset
        class_weights = compute_class_weight("balanced",
                classes=np.arange(163),
                y=np.array(targets))
        class_weights = torch.from_numpy(class_weights).float().to(accelerator.device)
        lossFN = nn.CrossEntropyLoss(weight=class_weights)
    else:
        lossFN = nn.CrossEntropyLoss()


    # Synchronize batch normalization
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Prepare for distributing with accelerate
    model, train_dl, val_dl, opt, scheduler = accelerator.prepare(
                model, train_dl, val_dl, opt, scheduler)

    # Save initial state
    accelerator.save_state(safe_serialization=False)
    #(output_dir=checkpoints_folder)

    # Calculate number of training and validation steps
    trainSteps = math.ceil(len(train_dl.dataset) / args.batch_size)
    valSteps = math.ceil(len(val_dl.dataset) / args.batch_size)


    # Store training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }


    # Training loop
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()  # Put model in training mode
        totalTrainLoss = 0  # Set the losses to 0 for the current epoch
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0

        # Loop over the training set
        for(x,y) in train_dl:

            # Forward pass and training loss
            pred = model(x)   # Make a prediction for x
            loss = lossFN(pred, y)  # Calculate the loss

            # Backpropagation and update weights
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            # Update total training loss and number of correct predictions
            preds, ys = accelerator.gather((pred,y))
            cor_preds = preds.argmax(1) == ys
            trainCorrect += cor_preds.sum()
            losses = accelerator.gather(loss)
            totalTrainLoss += losses.sum()

        # Evaluation
        with torch.no_grad():
            model.eval()  # Put model to evaluation mode

            # Loop over validation set
            for (x,y) in val_dl:
                pred = model(x)
                preds, ys = accelerator.gather((pred,y))
                cor_preds = preds.argmax(1) == ys
                valCorrect += cor_preds.sum()
                loss = lossFN(pred,y)
                losses = accelerator.gather(loss)
                totalValLoss += losses.sum()

        # Calculate average losses --> adapt to number of pictures in each batch
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # Calculate training and validation accuracy
        trainCorrect = trainCorrect/len(train_dl.dataset)
        valCorrect = valCorrect/len(val_dl.dataset)

        # Update training history
        history['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
        history['train_acc'].append(trainCorrect)
        history['val_loss'].append(avgValLoss.cpu().detach().numpy())
        history['val_acc'].append(valCorrect)

        endTime = time.time()
        # Print some statistics
        if accelerator.is_main_process:
            print("Epoch: {}/{}".format(epoch+1, epochs))
            print("Training time: {:.2f}s".format(endTime-startTime))
            #print("Learning rate: {}".format(scheduler.get_last_lr()[0]))
            print("Train loss: {:.4f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
            print("Validation loss: {:.4f}, Validation accuracy: {:.4f}".format(avgValLoss, valCorrect))

        #accelerator.wait_for_everyone()  # Just to make sure after evaluation
        accelerator.save_state(safe_serialization=False)
        #(output_dir=checkpoints_folder)

            # TODO: Add early stopping with target_accuracy and patience
            # Usually works with one process checking the condition and
            # breaking out of the training loop; otherwise use breakpoints

        # Update learning rate
        scheduler.step(avgValLoss)


    endTime = time.time()
    if accelerator.is_main_process:
        print("Total training time: {:.2f}s".format(endTime-startTime))
        # Save the complete training history
        np.save(results_folder + "/model_history.npy", history)

    # Save the complete model -> error because CUDA_HOME does not exist
    #accelerator.wait_for_everyone()
    accelerator.save_model(model, f"{results_folder}/final_model.bin")

    accelerator.end_training()


if __name__ =="__main__":
    worker(args)

