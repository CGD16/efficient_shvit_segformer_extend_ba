This paper can be found here: [PDF](https://git5.cs.fau.de/uw61ehod/shvit_segformer_ba/-/blob/2c6c2b004ab9b61702f9564de87c08095bf82ece/LaTeX/ChangGengDrewes_SHViT_Segformer_BA.pdf)


To facilitate the execution of Linux-based software and tools in a Windows environment,
Windows Subsystem for Linux (WSL) was set up on a Windows 11 machine. The following
steps outline the process of setting up WSL, installing Miniconda, and creating a virtual
environment to run PyTorch 2.60[^1] with GPU support.

[^1]: The PyTorch version 2.5.1 is also working.

After downloading and installing Ubuntu 24.04 using Microsoft Store the following steps
have to be carried out. Before step 4 is executed Miniconda had to be downloaded[^2]:

[^2]: https://www.anaconda.com/download/success

```
1 sudo apt update -y
2 sudo apt upgrade -y
3 cd /mnt/c/Users/USERNAME/Downloads/
4 bash Miniconda3-latest-Linux-x86_64.sh
5 cd ~
6 ~/miniconda3/bin/conda init bash
7 ~/miniconda3/bin/conda init zsh
8 exit
```

The installation of PyTorch is very easy and does not require additional CUDA drivers or other dependencies, unlike TensorFlow. PyTorch comes with built-in CUDA support[^3] meaning it can be simply installed using pip, and it will automatically handle the necessary CUDA libraries.


[^3]: https://pytorch.org/get-started/locally/


```
9 conda create -n torch260 python=3.12
10 conda activate torch260
11 pip3 install torch torchvision torchaudio --index-url
https://download.pytorch.org/whl/cu126
12 conda install -c conda-forge opencv matplotlib tqdm seaborn pandas plotly lightning
13 pip install torchsummary torchviz pydicom slicerio unfoldNd vedo
14 sudo apt-get install graphviz
15 cd /mnt/c/Users/USERNAME/Documents/Python/shoes_segformer/Software/
16 pip install PythonTools-3.7.0-py2.py3-none-any.whl
17 conda install conda-forge::libsqlite --force-reinstall
18 conda install conda-forge::sqlite --force-reinstall
```

After setting up PyTorch, essential libraries like `matplotlib`, `tqdm`, `seaborn`, etc. needed
for running the SegFormer scripts have to be installed. Additionally, `pydicom` and `slicerio`
were included for handling the annotation files for the shoe files. To extract data from the
original shoe volume data from the files in `.rek` dataformat, the `PythonTools` software was
required. While running the scripts in VSCode, the kernel occasionally crashed. This issue was resolved by
reinstalling the `sqlite` and `libsqlite` packages at the end[^4].

[^4]: https://stackoverflow.com/a/79484466/27900239















