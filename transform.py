import torchvision
import cv2
import numpy as np
class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample
class RandomGaussianNoise:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            noise = np.random.normal(self.mean, self.std, sample.shape)
            sample = sample + noise
        return sample
# ORL
class Transforms:
    def __init__(self, size, s=1.0, mean=None, std=None, blur=False, noise=False):
        self.train_transform = [
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0,0.8 * s, 0, 0)], p=0.5),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=23))

        if noise:
            self.train_transform.append(RandomGaussianNoise(mean=0, std=0.5))  # Adjust std as needed

        self.train_transform.append(np.array)

        self.test_transform = [
            torchvision.transforms.ToTensor(),
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)

        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        X=np.array(x)
        X = np.expand_dims(X, axis=0)

        train_transform = np.expand_dims(self.train_transform(x), axis=0)
        return X,train_transform