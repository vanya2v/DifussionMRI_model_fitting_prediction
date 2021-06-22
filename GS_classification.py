import sys
import pandas as pd
import monai
import glob
import nibabel as nib
import logging
import os
import shutil
import tempfile
import collections

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from monai.config import print_config
from monai.metrics import compute_roc_auc
from torch.utils.tensorboard import SummaryWriter
from monai.data import NiftiDataset
from torch.utils.data import DataLoader
from monai.losses import FocalLoss
from monai.networks.nets import densenet121, densenet169,densenet201,densenet264,se_resnet101,senet154,se_resnet152,se_resnet50
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score,roc_auc_score,roc_curve, auc,f1_score,cohen_kappa_score


from monai.transforms import (
    AddChannel,
    NormalizeIntensity,
    Compose,
    LoadPNG,
    RandFlip,
    RandRotate,
    RandZoom,
    RandAffine,
    Resize,
    ScaleIntensity,
    CropForeground,
    ToTensor,
)
from monai.utils import set_determinism
set_determinism(seed=0)

#list of of your patient here
patID = pd.read_excel('/SAN/medic/Verdict/experiment3/patID_44.xlsx') #patient ID list
gsID = pd.read_excel('/SAN/medic/Verdict/experiment3/gg_44F.xlsx') #Gleason score (labels)

#list of paraemtric maps from model fitting (in our case : VERDICT with compensated relaxation)
pmaps = ['fvasc', 'fic', 'fees', 'R', 'Cellularity', 'Dees', 'T2vasc_ees', 'T2ic', 'T1']


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    for m in range(4,5):#choose only FIC (or which maps you want to train with)
        print(pmaps[m])
        images = []
        labels = []
        for pat_ID in range(0, 44):

            subject_name = patID.values[pat_ID][0]
            train_images = sorted(glob.glob("/SAN/medic/Verdict/experiment3/"+subject_name+"/RVerdict_MLP/"+pmaps[m]+"_lesion*.nii.gz"))
            images.extend(train_images)

        for pat_ID in range(0, 64):
            gscore = gsID.values[pat_ID][0]

            if (gscore == 'No Cancer'):
                labels.extend('0')
            elif (gscore == '3+3'):
                labels.extend('1')
            elif (gscore == '3+4'):
                labels.extend('2')
            elif (gscore == '4+3'):
                labels.extend('3')
            else:
                labels.extend('4')


        lbl = np.array(labels, dtype=np.int64)

        # Data augmentation
        train_transforms = Compose([NormalizeIntensity(),ScaleIntensity(0,1),AddChannel(), CropForeground(select_fn=lambda x: x > 0, margin=0),RandAffine(prob=0.5),RandRotate(),Resize((96, 96)),ToTensor()])
        val_transforms = Compose([NormalizeIntensity(),ScaleIntensity(0,1),AddChannel(),CropForeground(select_fn=lambda x: x > 0, margin=0),Resize((96, 96)),ToTensor()])

        # Split data for validation
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        n =0
        acc=[]
        prec=[]
        f1=[]
        roc=[]

        y_true = list()
        y_pred = list()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the model 
        model =se_resnet50(spatial_dims=2, in_channels=1,num_classes=5).to(device)

        # Get the split of train and test datasets and training
        for train_index, test_index in sss.split(images,lbl):
            X_train, X_test =np.array(images)[train_index.astype(int)],np.array(images)[test_index.astype(int)],
            y_train, y_test = lbl[train_index], lbl[test_index]
            n=n+1

            train_ds = NiftiDataset(image_files=X_train, labels=y_train, transform=train_transforms)
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

            # create a validation data loader
            val_ds = NiftiDataset(image_files=X_test, labels=y_test, transform=val_transforms)
            val_loader = DataLoader(val_ds, batch_size=4, num_workers=4, pin_memory=torch.cuda.is_available

            loss_function = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), 1e-5)
            epoch_num =30
            val_interval = 1

            # start a typical PyTorch training
            val_interval = 2
            best_metric = -1
            best_metric_epoch = -1
            epoch_loss_values = list()
            metric_values = list()
            writer = SummaryWriter()
            for epoch in range(epoch_num):
                model.train()
                epoch_loss = 0
                step = 0
                for batch_data in train_loader:
                    step += 1
                    #print(batch_data)
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_len = len(train_ds) // train_loader.batch_size
                    writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)

                if (epoch + 1) % val_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        num_correct = 0.0
                        metric_count = 0
                        for val_data in val_loader:
                            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                            val_outputs = model(val_images)
                            value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                            
                            metric_count += len(value)
                            num_correct += value.sum().item()
                        metric = num_correct / metric_count
                        metric_values.append(metric)
                        if metric > best_metric:
                            best_metric = metric
                            best_metric_epoch = epoch + 1
                            #torch.save(model.state_dict(), "best_clf_lesion_fic_RV_fold"+str(n)+".pth")
                            model1=model
                            print("saved new best metric model")
                       
                        writer.add_scalar("val_accuracy", metric, epoch + 1)

            # Evaluate the model and compute performance metrics
            model1.eval()
            acc.append(best_metric)
            with torch.no_grad():
                metric_co=0
                num_corr=0
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    pred = model1(val_images).argmax(dim=1)
                    vval=(torch.eq(pred, val_labels))
                    metric_co += len(vval)
                    num_corr += vval.sum().item()

                    for i in range(len(pred)):
                        y_true.append(val_labels[i].item())
                        y_pred.append(pred[i].item())
                met = num_corr/ metric_co

            print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            writer.close()

        # Print all evaluation metrics    

        print('acc',np.mean(acc))
        print('kappa',cohen_kappa_score(y_true,y_pred, labels=[0,1,2,3,4],weights='quadratic'))
        print('f1',f1_score(y_true, y_pred,average='weighted'))
        print(classification_report(y_true, y_pred, labels=[0,1,2,3,4] , digits=4))
        print(confusion_matrix(y_true,y_pred,labels=[0,1,2,3,4]))

if __name__ == "__main__":
    main()
