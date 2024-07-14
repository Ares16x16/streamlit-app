# my-web

Dependencies:
python version: 3.12
Error for torch cu213: ```pipreqs --savepath=requirements.in && pip-compile```
So ```pip freeze``` is used, so it may only works on windows.

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

For Diffusion, xformers is needed:
https://github.com/facebookresearch/xformers

For YOLO:
https://pjreddie.com/darknet/yolo/
```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia ```
