import numpy as np
import scipy as sp
import torch
import matplotlib

def formProjMat(A):
    """
    Forms matrix which projects onto the column span of A
    Inputs:
     - A
    Outputs:
     - P_A = A (A^T A)^-1 A^T
    """
    return np.matmul(A,np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T))
    

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
  #return images
  #return np.add(images,1.)
  return (images+1.)/2.


def imsave(images, size, path):
  return matplotlib.pyplot.imsave(path, merge(images, size)) #sp.misc.imsave(path, merge(images, size))


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)