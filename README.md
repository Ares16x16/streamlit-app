# Streamlit app
This is my streamlit app, currently it only has 2 pages.
1. Image generation using stable diffusion with ControlNet
2. Human Detection on uploaded video
   
Dependencies: python version: 3.12

Error for torch cu213: ```pipreqs --savepath=requirements.in && pip-compile```
```pip freeze``` is used, it may only works on windows.

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

For Diffusion, xformers is needed:
https://github.com/facebookresearch/xformers

For YOLO:
https://pjreddie.com/darknet/yolo/
```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia ```
