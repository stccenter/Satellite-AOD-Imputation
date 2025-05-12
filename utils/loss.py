import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def generator_loss(generator, discriminator, X, Y, M, New_Y, H, ROI_ymask, ROI_xmask, month, season, lulc, alpha, beta):
    # Generator
    G_sample = generator(New_Y * ROI_ymask, M * ROI_ymask, X * ROI_xmask, month, season, lulc, ROI_ymask)
    G_sample = G_sample * ROI_ymask

    # Combine with original data
    Hat_New_Y = New_Y * M + G_sample * (1 - M)

    # Discriminator prediction
    D_prob = discriminator(Hat_New_Y.float() * ROI_ymask, H.float() * ROI_ymask)

    # Adversarial loss
    G_loss1 = -torch.mean((1 - M) * ROI_ymask * torch.log(D_prob + 1e-8))
    
    # MSE loss for observed data
    MSE_train_loss = torch.mean((M * ROI_ymask * New_Y - M * ROI_ymask * G_sample)**2) / torch.mean(M * ROI_ymask)
    MAE_train_loss = torch.mean(torch.abs(M * ROI_ymask * New_Y - M * ROI_ymask * G_sample)) / torch.mean(M * ROI_ymask)

    # Total Generator loss
    G_loss = G_loss1 + alpha * MSE_train_loss + beta * MAE_train_loss

    return G_loss, G_loss1, MSE_train_loss, MAE_train_loss, G_sample

def discriminator_loss(generator, discriminator, X, M, New_Y, H, ROI_ymask, ROI_xmask, month, season, lulc):
    # Generator
    G_sample = generator(New_Y * ROI_ymask, M * ROI_ymask, X * ROI_xmask, month, season, lulc, ROI_ymask)
    # Combine with original data
    Hat_New_Y = New_Y * M + G_sample * (1 - M)
    # Discriminator
    D_prob = discriminator(Hat_New_Y.float() * ROI_ymask, H.float() * ROI_ymask)
    # Flatten M to match the shape of D_prob
    # Loss calculation
    D_loss = -torch.mean((M * ROI_ymask) * torch.log(D_prob + 1e-8) + ((1 - M) * ROI_ymask) * torch.log(1. - D_prob + 1e-8))
    return D_loss, G_sample