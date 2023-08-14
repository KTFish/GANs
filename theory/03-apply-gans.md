# Course 3 - Apply Generative Adversarial Networks (GANs)

- Week 1: GANs for Data Augmentation and Privacy
- Week 2: Image-to-Image Translation with Pix2Pix
- Week 3: Unpaired Translation with CycleGAN
  ![Congratulations XD](image-90.png)

**Downstream task** - depends on the output of a previous task or process. It involves applying the pre-trained model's knowledge to a new problem.

## 1. Data Augmentation

- Use GANs for data augmentation if real images are to rare.
- ...
- Paper: [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719) (Cubuk, Zoph, Shlens, and Le, 2019)

| Pros                                     | Cons                                               |
| ---------------------------------------- | -------------------------------------------------- |
| Generation of labeled examples.          | Not useful if the generator overfits to real data. |
| Better than hand-crafted syntetic data.  | Limited diversity.                                 |
| Improve downstream model generalization. |                                                    |

### Courseras resources related to Data Augmentation

- With smaller datasets, GANs can provide useful data augmentation that substantially [improve classifier performance](https://arxiv.org/abs/1711.04340).
- You have one type of data already labeled and would like to make predictions on [another related dataset for which you have no labels](https://www.nature.com/articles/s41598-019-52737-x). (You'll learn about the techniques for this use case in future notebooks!)
- You want to protect the privacy of the people who provided their information so you can provide access to a [generator instead of real data](https://www.ahajournals.org/doi/full/10.1161/CIRCOUTCOMES.118.005122).
- You have [input data with many missing values](https://arxiv.org/abs/1806.02920), where the input dimensions are correlated and you would like to train a model on complete inputs.
- You would like to be able to identify a real-world abnormal feature in an image [for the purpose of diagnosis](https://link.springer.com/chapter/10.1007/978-3-030-00946-5_11), but have limited access to real examples of the condition.

## 2. Image-to-Image Translation

> _Translating one possible representation of a
> scene into another, given sufficient training data._ (definition from Image-to-Image Translation with Conditional Adversarial Networks paper).

![Image-to-Image summary](image-91.png)

## 3. Pix-2-Pix

![why pixt 2 pix?](image-107.png)

### Pix-to-Pix Generator

- Generator inspired with U-Net
  ![generator](image-92.png)

![Discriminator](image-93.png)

### Patch GAN

![Alt text](image-94.png)

### U-Net Framework

- **encoder-decoder** framework.

![U-Net Framework](image-95.png)

- **skip connections** - allow information flow to the decoder and improve gradient flow to the encoder. They also help prevent the vanishing gradient problem.

### Pixel Distance Loss Term

![outline](image-96.png)
![Additional Loss term](image-97.png)
![pixel loss](image-98.png)
![generator loss](image-99.png)
![summary](image-100.png)
![Loss - Screen from paper, could be helpful](image-106.png)

![pix2pix](image-101.png)
![pix2pix2](image-102.png)

![summary](image-103.png)

### Improvements of Pix2Pix

![outline](image-104.png)

- paper Pix2PixHD
- paper GauGAN
  ![summary](image-105.png)

## 4.
