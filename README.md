# Stability API Extension for Automatic1111 WebUI (Beta Release)

![image](https://user-images.githubusercontent.com/26013475/221394848-b05478e7-5512-485e-a41a-d8eca5280dc4.png)

The Stability API Extension for Automatic1111 WebUI enables users to generate Stable Diffusion images via the Stability API instead of on a local GPU.

## Features

- **txt2img**: Generate images from text descriptions

- **img2img**: Generate images from input images

- **inpainting**: Generate over specific portions of an image -use an inpainting model for best results!

- **Batch Image Requests**: Generate large batches of images at a time

- **Upscale ESRGAN x2Plus**: Upscale images to 2x their size (use via img2img)

- **X/Y/Z plot support**: Generate plots to compare settings and outputs

## Benefits

- Currently, the only way to test SDXL-beta with Automatic1111

- Fast and cheap inference

- Trying out the newest models from Stability

- Avoids the hassle of managing your own GPU resources while still receiving high-quality images

- No need to store large models and checkpoints on your personal computer

- Convenient integration with Automatic1111 WebUI

## Installation

1. Install AUTOMATIC1111 webui from https://github.com/AUTOMATIC1111/stable-diffusion-webui

2. In the `Extensions > Install from URL` tab, paste in: `https://github.com/Stability-AI/webui-stability-api/`

![SAPI-install-0](https://user-images.githubusercontent.com/100188076/227592927-e4b9117f-0e7f-462a-9348-7f2fc28b2a30.jpg)

3. Back in the `Extensions > Installed` tab, restart the UI:

![SAPI-install-1](https://user-images.githubusercontent.com/100188076/221432363-552d7b3b-4600-460e-b2e7-226a25072a26.jpg)

4. Generate a Stability API key at https://beta.dreamstudio.ai/membership?tab=apiKeys 

[![SAPI key](https://user-images.githubusercontent.com/100188076/221430957-9cbe0f3e-21a8-4bc0-8d27-d725499a0038.jpg)](https://beta.dreamstudio.ai/membership?tab=apiKeys)
  
5. In the `Stability API Settings` tab. Your account information and available credits will show up on the settings page.

![SAPI-key-2](https://user-images.githubusercontent.com/100188076/221431058-04e98612-0dbe-449a-90bb-cea1aa0a45df.jpg)

6. Refresh your checkpoints to see all the models you can use:

![checkpoints](https://user-images.githubusercontent.com/26013475/221395323-2bca27c6-b82a-4910-975f-903bba85ea39.png)

7. Enjoy creating without making your GPU going BRRRRR!

## Known Issues

1. May have compatability issues with other extensions
