# Image-Denoising-using-AutoEncoder-Paradigm


In this project, I am using the Encoder-Decoder Paradigm to denoise images. The encoder-decoder architecture, commonly used in deep learning, is particularly effective for image denoising tasks. The encoder compresses the input image into a lower-dimensional representation, capturing its essential features while removing noise. The decoder then reconstructs the image from this representation, restoring the original details and further eliminating any remaining noise.

Here is a sample of how the images look before and after the denoising process:

Before

  
  ![aug_267_9](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/c2abbd67-0de3-451c-bf35-d4caacc26f99)    ![aug_2246_7](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/a3ed133a-6864-4079-99ec-7c8762a0d77b)   ![aug_446_7](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/b761973f-47a2-4f9e-82cc-74c0e57859f5)    ![aug_834_4](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/a183e91d-f1d5-4e78-a21e-5e948195f33b)    ![aug_1489_0](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/6267d835-d0d0-4c4e-8e44-7c4956e5d786)


After


![clean_19_9](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/120008e6-9257-4402-ad9f-5f3cfbca2e0e)   ![clean_52_7](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/72d9bf45-f715-49df-8418-569438624aba)   ![clean_64_4](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/82b6cf53-860e-4619-8f0e-20cd2aacf965)   ![clean_217_4](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/e10e63d3-ccb4-4362-856d-003a7212916f)   ![clean_326_0](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/0f2b39bd-e6ec-46ca-8072-fc70b3308e39)



Data Preprocessing :
    For data preprocessing, I was using Difference of Gaussian beacuse this method is more sophisticated as it smooths the image while preserving edges based on both spatial and intensity information, often outperforming DoG in terms of edge preservation but at the cost of higher computational complexity.


In this project, I have used mainly two Autoencoders:

  1. Classical Autoencoder with ResNet Design
  2. Variational Autoencoder


Metric used in this process SSI score(structure similarity score ,a method for predicting the perceived quality of digital television and cinematic pictures, as well as other kinds of digital images and videos)

Results:
  1. Resnet AutoEncoder ssim score :  0.7043934555175253
  2. VAE AutoEncoder ssim score :  0.8554861151928369

After that I have plot the T-SNE plots of the entire dataset

![image](https://github.com/coolLaksh/Image-Denoising-using-Encoder-Decoder-Paradigm/assets/116641733/2049b176-131b-418e-bf60-7bb1dc13d157)


You can acesss the dataset from here : https://drive.google.com/file/d/17nsLFjJxfcf9UPoNjeBjovfi4AL7EGKB/view?usp=sharing


  
