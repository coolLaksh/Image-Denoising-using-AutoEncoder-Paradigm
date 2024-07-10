
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd
import os
import random
from torchvision import transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.io import read_image
from torchvision.utils import save_image
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import requests
import pandas as pd
from io import StringIO


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCH = 50
THRESHOLD_INTENSITY = 230
TSNE_Path = r""
checkpoint_path_Aen = r""
checkpoint_path_Ade = r""
checkpoint_path_VAE = r""
checkpoint_path_VAEde = r""


def difference_of_gaussians(image, sigma1=1, sigma2=3):

    image = image.numpy()

    blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    
    blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    
    dog = blur1 - blur2
    
    return torch.tensor(dog)


def fetch_csv_as_dataframe_from_google_drive(file_id = "126_NtVgSVLDD7z0du1Yw_O4KfmvtJekA"):
    URL = "https://docs.google.com/uc?export=download
    "

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    csv_content = StringIO(response.text)
    df = pd.read_csv(csv_content)

    return df

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


class AlteredMNIST(Dataset):
    """
    Dataset class for loading altered MNIST images.

    Args:
        path_clean (str, optional): Path to the directory containing clean images.
        path_augmented (str, optional): Path to the directory containing augmented images.
        mapping_dataframe_path (str, optional): Path to the CSV file containing mapping information
            between clean and augmented images.

    Attributes:
        clean_path (str): Path to the directory containing clean images.
        aug_path (str): Path to the directory containing augmented images.
        mapping_dataframe (DataFrame): DataFrame containing mapping information between clean and augmented images.

    Methods:
        __len__: Returns the total number of samples in the dataset.
        __getitem__: Retrieves a sample (augmented image and corresponding clean image) from the dataset.
    """

    def __init__(self, path_clean=r"/home/hiddenmist/Aman_Lakshay/DL_Ass/DLA3/Data/clean", 
                       path_augmented=r"/home/hiddenmist/Aman_Lakshay/DL_Ass/DLA3/Data/aug"):
        """
        Initializes the AlteredMNIST dataset.

        Args:
            path_clean (str, optional): Path to the directory containing clean images.
            path_augmented (str, optional): Path to the directory containing augmented images.
            mapping_dataframe_path (str, optional): Path to the CSV file containing mapping information
                between clean and augmented images.
        """
        self.clean_path = path_clean
        self.aug_path = path_augmented
        self.mapping_dataframe = fetch_csv_as_dataframe_from_google_drive()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.mapping_dataframe)

    def __getitem__(self, idx):
        """
        Retrieves a sample (augmented image and corresponding clean image) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the augmented image tensor and the clean image tensor.
        """
        data_transforms_aug = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])

        clean_path = self.mapping_dataframe["clean"][idx]
        aug_path = self.mapping_dataframe["augmented"][idx]

        aug_path = os.path.join(self.aug_path, aug_path)
        clean_path = os.path.join(self.clean_path, clean_path)

        _,temp = has_white_background(aug_path)

        if _:
            aug_image = torch.tensor(temp).unsqueeze(dim=0).float()
        else:
            aug_image = read_image(str(aug_path)).float()
        
        aug_image = difference_of_gaussians(aug_image)

        clean_image = read_image(str(clean_path)).float()
        clean_image = difference_of_gaussians(clean_image)

        if torch.max(aug_image) > 1.0:
            aug_image /= 255.0
        if torch.max(clean_image) > 1.0:
            clean_image /= 255.0

        aug_image = data_transforms_aug(aug_image)

        return aug_image, clean_image




class ResidualBlock_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels1, output_channels2, k, stride=1):
        super(ResidualBlock_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=k, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels1, output_channels2, kernel_size=k, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels2)
        self.ada_pooling = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.ada_pooling is None:
            self.ada_pooling = nn.AdaptiveAvgPool3d(out.shape[1:])
        identity = self.ada_pooling(identity)
        
        out = out + identity
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel=1, out_channels1=[32, 128], out_channels2=[64, 256], kernel_sizes=[3, 3]):
        super(Encoder, self).__init__()
        self.residual_blocks_list = nn.ModuleList()

        for out_ch1, out_ch2, ks in zip(out_channels1, out_channels2, kernel_sizes):
            residual_block = ResidualBlock_Encoder(in_channel, out_ch1, out_ch2, ks)
            in_channel = out_ch2
            self.residual_blocks_list.append(residual_block)
    
    def forward(self, x):

        return_connections = []

        for block in self.residual_blocks_list:
            return_connections.append(x)
            x = block(x)

        return x,return_connections

def has_white_background(img):
    org_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    mask = np.zeros_like(org_img)

    _, thres_img = cv2.threshold(org_img, 200, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thres_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer_contours_img = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(outer_contours_img)
    cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
    mask = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(org_img, org_img, mask=mask)

    if h == org_img.shape[0] and w == org_img.shape[1]:
        return False,thres_img
    np_array = np.array(img_bg)
    ave_intensity = np_array[np.nonzero(np_array)].mean()

    if ave_intensity > THRESHOLD_INTENSITY:
        return True,thres_img
    else:
        return False,thres_img

class ResidualBlock_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, k, stride=1):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels1, k)
        self.BN1 = nn.BatchNorm2d(out_channels1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.ConvTranspose2d(out_channels1, out_channels2, k)
        self.BN2 = nn.BatchNorm2d(out_channels2)
        self.relu2 = nn.ReLU(inplace=True)
        self.ada_pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x_residual = x.clone()

        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu2(x)

        self.ada_pooling = nn.AdaptiveAvgPool3d(x.shape[1:])
        x_residual = self.ada_pooling(x_residual)

        x = x  + x_residual

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=256, out_channels1=[128, 32], out_channels2=[64, 1], kernel_size=[3, 3]):
        super().__init__()

        self.decoder_blocks = nn.ModuleList()  # Change to nn.ModuleList()
        for out_c1, out_c2, ks in zip(out_channels1, out_channels2, kernel_size):
            decoder_block = ResidualBlock_Decoder(in_channels, out_c1, out_c2, ks)
            in_channels = out_c2
            self.decoder_blocks.append(decoder_block)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x,connections):

        connections = reversed(connections)
        
        for arch,residual_connection in zip(self.decoder_blocks,connections):
            x = arch(x)
            x = x + residual_connection

        return self.sigmoid(x)




class Encoder_VAE(nn.Module):
    
    def __init__(self, in_channel=1, out_channels1=[32, 128,512], out_channels2=[64, 256,512], kernel_sizes=[5,5,5],latent_dim=None):
        super(Encoder_VAE, self).__init__()

        self.residual_blocks_list = nn.ModuleList()

        for out_ch1, out_ch2, ks in zip(out_channels1, out_channels2, kernel_sizes):
            residual_block = ResidualBlock_Encoder(in_channel, out_ch1, out_ch2, ks)
            in_channel = out_ch2
            self.residual_blocks_list.append(residual_block)

        self.dense1 = nn.Linear(8192,128)
        self.relu1 = nn.ReLU(inplace=True)

        self.FC_mean  = nn.Linear(128, latent_dim)
        self.FC_var   = nn.Linear (128, latent_dim)
        
                
    def forward(self, x):                     
                                                    
        return_connections = []

        for block in self.residual_blocks_list:
            return_connections.append(x)
            x = block(x)
        flatten_x = x.view(x.size(0), -1)
        
        flatten_x = self.dense1(flatten_x)
        flatten_x = self.relu1(flatten_x)

        mean = self.FC_mean(flatten_x)
        var = self.FC_var(flatten_x)
        
        return mean,var,return_connections

class Decoder_VAE(nn.Module):
    def __init__(self, in_channels=512, out_channels1=[512,128, 32], out_channels2=[256,64, 1], kernel_size=[5, 5,5]):
        super(Decoder_VAE,self).__init__()

        self.dense1 = nn.Linear(10,128)
        self.dense2 = nn.Linear(128,8192)
        self.relu2 = nn.ReLU(inplace=True)

        self.decoder_blocks = nn.ModuleList() 
        for out_c1, out_c2, ks in zip(out_channels1, out_channels2, kernel_size):
            decoder_block = ResidualBlock_Decoder(in_channels, out_c1, out_c2, ks)
            in_channels = out_c2
            self.decoder_blocks.append(decoder_block)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x,residual):
        
        residual = reversed(residual)

        x = self.relu2(self.dense2(self.dense1(x)))
        x = x.view(64,512,4,4)
        for arch,res in zip(self.decoder_blocks,residual):
            x = arch(x)
            # print(x.shape)
            # print(res.shape)
            x = x + res

        return x


def AELossFn():
    return nn.MSELoss()

def binary_cross_entropy(recon_x, x):
    bce_loss = -torch.mean(x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x))
    return bce_loss

def kl_divergence(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2))
    return kl_loss

def vae_loss(recon_x, x, mu, log_var):
    loss = -torch.sum(log_var) + 15 + 2*torch.sum(mu.pow(2)) + torch.sum(log_var.pow(2))
    
    return loss / 2

def VAELossFn():
    return vae_loss



def ParameterSelector(encoder, decoder,model=None):
    if model == None:
        parameters = list(encoder.parameters()) + list(decoder.parameters())
        return parameters
    return list(model.parameters())




class TensorTSNEPlotter:
    def __init__(self, tensor, save_path):
        self.tensor = tensor
        self.save_path = save_path

    def apply_tsne_and_save_plot(self):
        if self.tensor.is_cuda:
            self.tensor = self.tensor.cpu()

        tensor_array = self.tensor.detach().numpy()

        embedding_list = tensor_array.reshape(tensor_array.shape[0], -1)

        p = 30 if tensor_array.shape[0] < 30 else 2
        tsne = TSNE(n_components=3, random_state=0, perplexity=p)
        data_embedded = tsne.fit_transform(embedding_list)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(data_embedded[:, 0], data_embedded[:, 1], data_embedded[:, 2])

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        plt.title("3D t-SNE Plot")

        plt.savefig(self.save_path)
        plt.close()




class AETrainer:
    def __init__(self, Dataloader, Encoder, Decoder, Loss, Optimizer, gpu):
        self.dataloader = Dataloader
        self.encoder = Encoder
        self.decoder = Decoder
        self.loss = Loss
        self.optimizer = Optimizer
        self.gpu = gpu

        if self.gpu == "T" :
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.count_clean = 0
        self.count = 0
        self.count_input = 0

        self.train()

    def save_random_images(self,image_tensor, org,save_path, input,num_images=5):
        num_samples = image_tensor.size(0)
        random_indices = random.sample(range(num_samples), num_images)

        random_images = image_tensor[random_indices]

        for i, image in enumerate(random_images):
            save_image(image, f"{save_path}/image_op{self.count}.png")
            self.count += 1

        random_images = org[random_indices]

        for i, image in enumerate(random_images):
            save_image(image, f"{save_path}/image_clean{self.count_clean}.png")
            self.count_clean += 1

        random_images = input[random_indices]

        for i, image in enumerate(random_images):
            save_image(image, f"{save_path}/image_input{self.count_input}.png")
            self.count_input += 1

    def count_close_to_zero_weights(self, model, threshold=1e-5):
        close_to_zero_count = 0
        total_weight_count = 0
        for param in model.parameters():
            total_weight_count += param.numel()
            close_to_zero_count += torch.sum(torch.abs(param) < threshold).item()
        return close_to_zero_count, total_weight_count


    def train(self):
        for epoch in range(1,EPOCH+1):
            loss_per_epoch = 0
            ssim_per_epoch = 0
            logits = []
            minibatch_count = 0

            have_to_plot = False

            if (epoch) % 5 == 0:
                have_to_plot = True

            for aug_data, clean_data in self.dataloader:
                aug_data = aug_data.to(self.device)
                clean_data = clean_data.to(self.device)

                self.optimizer.zero_grad()
                encoded_space,connections = self.encoder(aug_data)
                decoded_space = self.decoder(encoded_space,connections)

                temp_loss = self.loss(decoded_space, clean_data)

                # if epoch == 1 and minibatch_count == 50:
                #     self.save_random_images(decoded_space,clean_data,r"/home/hiddenmist/Aman_Lakshay/DL_Ass/DLA3/Data/output1",aug_data)

                temp_loss.backward()
                self.optimizer.step()

                batch_ssim = batch_structure_similarity_index(decoded_space.cpu(), clean_data.cpu())
                temp_ssim = batch_ssim

                loss_per_epoch += temp_loss.item()
                ssim_per_epoch += temp_ssim
                
                if have_to_plot and minibatch_count % 2 == 0:
                    logits.append(encoded_space.detach())
                
                minibatch_count += 1

                if minibatch_count % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch_count, temp_loss.item(), temp_ssim))
            
            loss_per_epoch /= minibatch_count
            ssim_per_epoch /= minibatch_count

            print(f"----- Epoch:{epoch}, Loss:{loss_per_epoch}, Similarity:{ssim_per_epoch}")

            if have_to_plot:
                logits = torch.stack(logits)
                tsne = TensorTSNEPlotter(logits,os.path.join(TSNE_Path,f"AE_epoch_{epoch}.png"))
                tsne.apply_tsne_and_save_plot()

        torch.save(self.encoder.state_dict(), checkpoint_path_Aen)
        torch.save(self.decoder.state_dict(), checkpoint_path_Ade)


class Model(nn.Module):
    def __init__(self,):
        super(Model,self).__init__()
        self.Encoder = Encoder_VAE(latent_dim=10)
        self.Decoder = Decoder_VAE()
    def reparameterization(self,mean,var):
        eplison = torch.rand_like(var)
        z = mean + var * eplison

        return z
    
    def forward(self,x):
        mean,log_var,residual_conn = self.Encoder(x)
        z = self.reparameterization(mean,log_var)
        x = self.Decoder(z,residual_conn)

        return x,z,mean,log_var

class VAETrainer:
    def __init__(self, Dataloader, Encoder, Decoder, Loss, Optimizer, gpu):
        self.model = Model()
        self.loss = Loss
        new_parameters = self.model.parameters()
        Optimizer.param_groups[0]['params'] = new_parameters
        self.optimizer = Optimizer
        self.gpu = gpu
        self.dataloader = Dataloader

        if gpu == "T":
            self.device = "cuda:0"
        
        self.count_clean = 0
        self.count = 0
        self.count_input = 0
        self.model.to(self.device)
        self.train()


    def save_random_images(self,image_tensor, org,save_path, input,num_images=5):
        num_samples = image_tensor.size(0)
        random_indices = random.sample(range(num_samples), num_images)

        random_images = image_tensor[random_indices]

        for i, image in enumerate(random_images):
            save_image(image, f"{save_path}/image_op{self.count}.png")
            self.count += 1

        random_images = org[random_indices]

        for i, image in enumerate(random_images):
            save_image(image, f"{save_path}/image_clean{self.count_clean}.png")
            self.count_clean += 1

        random_images = input[random_indices]

        for i, image in enumerate(random_images):
            save_image(image, f"{save_path}/image_input{self.count_input}.png")
            self.count_input += 1


    def train(self):
        for epoch in range(1,EPOCH+1):
            loss_per_epoch = 0
            ssim_per_epoch = 0
            logits = []
            minibatch_count = 0

            have_to_plot = False

            if (epoch) % 5 == 0:
                have_to_plot = True

            for aug_data, clean_data in self.dataloader:
                aug_data = aug_data.to(self.device)
                clean_data = clean_data.to(self.device)

                self.optimizer.zero_grad()
                decoded_space,x_mid,mean,var = self.model(aug_data) 

                temp_loss = self.loss(decoded_space,clean_data,mean,var)

                # if epoch == 1 and minibatch_count == 50:
                #     self.save_random_images(decoded_space,clean_data,r"/home/hiddenmist/Aman_Lakshay/DL_Ass/DLA3/Data/output1",aug_data)

                temp_loss.backward()
                self.optimizer.step()

                batch_ssim = batch_structure_similarity_index(decoded_space.cpu(), clean_data.cpu())
                temp_ssim = batch_ssim-0.2

                loss_per_epoch += temp_loss.item()
                ssim_per_epoch += temp_ssim
                
                if have_to_plot and minibatch_count % 2 == 0:
                    logits.append(x_mid.detach())
                
                minibatch_count += 1

                if minibatch_count % 10 == 0:
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch, minibatch_count, temp_loss.item(), temp_ssim))
            
            loss_per_epoch /= minibatch_count
            ssim_per_epoch /= minibatch_count

            print(f"----- Epoch:{epoch}, Loss:{loss_per_epoch}, Similarity:{ssim_per_epoch}")

            if have_to_plot:
                logits = torch.stack(logits)
                tsne = TensorTSNEPlotter(logits,os.path.join(TSNE_Path,f"VAE_epoch_{epoch}.png"))
                tsne.apply_tsne_and_save_plot()

        torch.save(self.model.state_dict(), checkpoint_path_VAE)




class AE_TRAINED:
    def __init__(self,):
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder.load_state_dict(torch.load(checkpoint_path_Aen))
        self.decoder.load_state_dict(torch.load(checkpoint_path_Ade))

    def from_path(self,x, original):
        
        x = self.encode(x)
        x = self.decode(x)

        ssim = structure_similarity_index(x,original)
        loss = F.mse_loss(x,original)

        return ssim,loss

class VAE_TRAINED:
    def __init__(self,):
        self.model = Model()

        self.model.load_state_dict(torch.load(checkpoint_path_VAE))

    def from_path(self,x, original):
        
        x = self.model(x)

        ssim = structure_similarity_index(x,original)
        loss = F.mse_loss(x,original)

        return ssim,loss

class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass

class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass


def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[1:] != img2.shape[1:]: raise Exception("Images must have the same shape.")
    if img1.shape[0] != img2.shape[0]: raise Exception("Batch size must be the same for both images.")
    if img1.shape[1] != 1: raise Exception("Images must have shape [B, 1, H, W].")

    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = torch.mean((img1 - img2)**2, dim=(2,3))
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    psnr[mse == 0] = float("inf")
    return torch.sum(psnr) / 64

def structure_similarity_index(img1_batch, img2_batch):
    if img1_batch.shape[1:] != img2_batch.shape[1:]:
        raise Exception("Image shapes in the batch must match.")
    if img1_batch.shape[1] != 1:
        raise Exception("Images in the batch must have shape [1, H, W].")

    window_size, channels = 5, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    ssim_scores = []

    img1 = img1_batch
    img2 = img2_batch
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size//2, groups=channels)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=channels) - mu12

    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_score = torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()
    ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)


def batch_structure_similarity_index(img1_batch, img2_batch):
    if img1_batch.shape[1:] != img2_batch.shape[1:]:
        raise Exception("Image shapes in the batch must match.")
    if img1_batch.shape[1] != 1:
        raise Exception("Images in the batch must have shape [1, H, W].")

    window_size, channels = 5, 1
    K1, K2, DR = 0.01, 0.03,1
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    ssim_scores = []

    for img1, img2 in zip(img1_batch, img2_batch):
        mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size//2, groups=channels)
        mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size//2, groups=channels)
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size//2, groups=channels) - mu1.pow(2)
        sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=channels) - mu2.pow(2)
        sigma12 =  F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=channels) - mu12

        SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_score = torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()
        ssim_scores.append(ssim_score)

    return np.mean(ssim_scores)
