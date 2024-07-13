# my-web
my web

command to generate requirements.txt:
```
pipreqs --savepath=requirements.in && pip-compile
```
This error for torch cu213 so pip freeze is used.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Have to install xFormers for diffusion model:
https://github.com/facebookresearch/xformers