# the following code is build upon https://github.com/ahujak/WSRL
import numpy as np
import torch
import pygame
from pygame import gfxdraw
 


SCREEN_DIM = 64
n_balls = 1
COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
]

def circle(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=0.0,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset

    gfxdraw.aacircle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
    gfxdraw.filled_circle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )


def draw_scene(z):
    ball_rad = 0.04
    screen_dim = 64
    surf = pygame.Surface((screen_dim, screen_dim))
    screen = pygame.display.set_mode((screen_dim, screen_dim))
    surf.fill((255, 255, 255))
    if z.ndim == 1:
        z = z.reshape((1, 2))
    for i in range(z.shape[0]):
        if all(z[i,:]>0):
            circle(
                z[i, 0],
                z[i, 1],
                surf,
                color=COLOURS_[i],
                radius=ball_rad,
                screen_width=screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
    surf = pygame.transform.flip(surf, False, True)
    
    screen.blit(surf, (0, 0))
    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
    )


def generate_ball_dataset(n_balls, means, Sigma, masks, mask_value, sample_num_per_group):
    Z=[]
    X=[]
    for mask in masks:
        for j in range(sample_num_per_group):
            while True:
                z = np.random.multivariate_normal(np.zeros(2), Sigma, n_balls)
                z = z+means
                if np.all(z <= 0.9) and np.all(z >= 0.1):
                    break
            
            for l1 in range(n_balls):
                for l2 in range(2):
                    if mask[l1][l2]==0:
                        z[l1,l2]=mask_value[l1][l2]
            
            x = draw_scene(z)
            Z.append(z.flatten())
            X.append(x)
    return torch.Tensor(np.array(Z)), torch.Tensor(np.transpose(np.array(X), axes=(0,3,1,2)))

def generate_missing_ball_dataset(n_balls, means, Sigma, masks, mask_value, sample_num_per_group):
    Z=[]
    X=[]
    for mask in masks:
        for j in range(sample_num_per_group):
            while True:
                z = np.random.multivariate_normal(np.zeros(2), Sigma, n_balls)
                z = z+means
                if np.all(z <= 0.9) and np.all(z >= 0.1):
                    break
            
            for l1 in range(n_balls):
                for l2 in range(2):
                    if mask[l1][l2]==0:
                        z[l1,l2]=mask_value[l1][l2]
            
            x = draw_scene(z)
            Z.append(z[:,0].flatten())
            X.append(x)
    return torch.Tensor(np.array(Z)), torch.Tensor(np.transpose(np.array(X), axes=(0,3,1,2)))
  
