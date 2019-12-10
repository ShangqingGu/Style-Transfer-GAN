Description 
=========== 
This is project Style-Transfer-GAN (Project B) developed by team MatrixLoss composed of Shangqing Gu, Hejin Liu, Yi Hui Chen

Requirements 
============ 
No additional package needed when runing on dsmlp server
Please execute Original-paper-implementation/download_model.sh before running the notebook for section 1

Code organization 
================= 
Original-paper-implementation/* -- All the neural style transfer code for Section 1 of project B

demo.ipynb -- Run a demo of our code to see two samples of our style transfer results
CycleGAN-train-0.ipynb -- Run the GAN experiment 1 as described in Section 3 of project B
CycleGAN-train-1.ipynb -- Run the GAN experiment 2 as described in Section 3 of project B
CycleGAN-train-2.ipynb -- Run the GAN experiment 3 as described in Section 3 of project B
code/models.py -- Module implementing GAN model structures
code/optimizer.py -- Where we define loss functions and optimizer
code/utils.py -- Helper functions to hold pool buffers

Example for saved models:
assets/Exp0-photo2vangogh/saved_models/D__A_10.pth -- Descriminator A trained for 10 epoch by experiment 1
assets/Exp0-photo2vangogh/statistics/10 -- the statistics collected during train for 10th epoch by experiment 1

Example for dataset images we used to train or test:
datasets/photo2vangogh/trainA/10003197063_b94d37d2e8.jpg -- a training image in domain A (the landscape photo)
