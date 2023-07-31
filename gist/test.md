### Transpose Convolution `torch.nn.ConvTranspose2d`

> ⚠️ The name can be confussig. There is no transpose or real convolution used.
> A good explanation of this by [Shubham Singh](https://www.youtube.com/watch?v=U3C8l6w-wn0) on YouTube.

- Transpose convolution means to scalar-multiply a kernel by each pixel in an image.
- The dimensions of the result tensor of transpose convolution is greater than the source dimensions.
- It is used to upscale images.
- Takes the same parameters as standard convolution: `kernel_size`, `padding` and `stride`.

$$N_h = s_h(M_h - 1) + k - 2p$$
Where:

- $N_h$ - Number of pixels in output image.
- $M_h$ - Number of pixels in input image.
- $s_h$ - Stride (skipping parameter).
- $p$ - Padding.
- $k$ - Kernel size.

<details>
<summary>
<font size="3" color="green">
<b>Visual Example of Transpose Convolution</b>
</font>
</summary>
<img src=".\figures\transpose.png">

</details>

<img src=".\figures\transpose.png">
