# NeLF: Neural Light-transport Field for Single Portrait View Synthesis and Relighting

Official PyTorch Implementation of paper "NeLF: Neural Light-transport Field for Single Portrait View Synthesis and Relighting", EGSR 2021.

[Tiancheng Sun](http://kevinkingo.com/)<sup>1*</sup>, [Kai-En Lin](https://cseweb.ucsd.edu/~k2lin/)<sup>1*</sup>, [Sai Bi](https://sai-bi.github.io/)<sup>2</sup>, [Zexiang Xu](https://cseweb.ucsd.edu/~zex014/)<sup>2</sup>, [Ravi Ramamoorthi](https://cseweb.ucsd.edu/~ravir/)<sup>1</sup>

<sup>1</sup>University of California, San Diego, <sup>2</sup>Adobe Research

<sup>*</sup>Equal contribution

[Project Page](https://cseweb.ucsd.edu//~viscomp/projects/EGSR21NeLF/) | [Paper](https://cseweb.ucsd.edu//~viscomp/projects/EGSR21NeLF/assets/nelf_egsr.pdf) | [Pretrained models](https://drive.google.com/file/d/1fwYvPZPXlAhTM5w7jQSFfk9oYWGGN5ZX/view?usp=sharing) | [Validation data](https://drive.google.com/file/d/1W93cU97EFLmzAvbKLB3LUvo61RWx59kY/view?usp=sharing) | [Rendering script](https://github.com/ken2576/facescape_render)

## Requirements

### Install required packages

Make sure you have up-to-date NVIDIA drivers supporting CUDA 11.1 (10.2 could work but need to change ```cudatoolkit``` package accordingly)

Run

```
conda env create -f environment.yml
conda activate pixelnerf
```

The following packages are used:

* PyTorch (1.7 & 1.9.0 Tested)

* OpenCV-Python

* matplotlib

* numpy

* tqdm

OS system: Ubuntu 20.04

### Download CelebAMask-HQ dataset [link](https://github.com/switchablenorms/CelebAMask-HQ)

1. Download the dataset

2. Remove background with the provided masks in the dataset

3. Downsample the dataset to 512x512

4. Store the resulting data in ```[path_to_data_directory]/CelebAMask```

    Following this data structure

    ```
    [path_to_data_directory] --- data --- CelebAMask --- 0.jpg
                                       |              |- 1.jpg
                                       |              |- 2.jpg
                                       |              ...
                                       |- blender_both --- sub001
                                       |                |- sub002
                                       |                ...

    ```

### (Optional) Download and render FaceScape dataset [link](https://facescape.nju.edu.cn/)

Due to [FaceScape's license](https://facescape.nju.edu.cn/Page_FAQ/), we cannot release the full dataset. Instead, we will release our rendering script.

1. Download the dataset

2. Install Blender [link](https://www.blender.org/)

3. Run rendering script [link](https://github.com/ken2576/facescape_render)


## Usage

### Testing

0. Download [our pretrained checkpoint](https://drive.google.com/file/d/1fwYvPZPXlAhTM5w7jQSFfk9oYWGGN5ZX/view?usp=sharing) and [testing data](https://drive.google.com/file/d/1ZPVnK68veiJK1v0ZrRBt5F2AGLqFIO9e/view?usp=sharing). Extract the content to ```[path_to_data_directory]```.
    The data structure should look like this:
    ```
    [path_to_data_directory] --- data --- CelebAMask
                              |        |- blender_both
                              |        |- blender_view
                              |        ...
                              |- data_results --- nelf_ft
                              |- data_test --- validate_0
                                            |- validate_1
                                            |- validate_2
    ```

1. In ```arg/__init__.py```, setup data path by changing ```base_path```

2. Run ```python run_test.py nelf_ft [validation_data_name] [#iteration_for_the_model]```

    e.g. ```python run_test.py nelf_ft validate_0 500000```

3. The results are stored in ```[path_to_data_directory]/data_test/[validation_data_name]/results```

### Training

Due to [FaceScape's license](https://facescape.nju.edu.cn/Page_FAQ/), we are not allowed to release the full dataset. We will use validation data to run the following example.

0. Download [our validation data](https://drive.google.com/file/d/1W93cU97EFLmzAvbKLB3LUvo61RWx59kY/view?usp=sharing). Extract the content to ```[path_to_data_directory]```.
    The data structure should look like this:
    ```
    [path_to_data_directory] --- data --- CelebAMask
                              |        |- blender_both
                              |        |- blender_view
                              |        ...
                              |- data_results --- nelf_ft
                              |- data_test --- validate_0
                                            |- validate_1
                                            |- validate_2
    ```


    (Optional) Run rendering script and render your own data.

    Remember to change line 35~42 and line 45, 46 in ```arg/config_nelf_ft.py``` accordingly.

1. In ```arg/__init__.py```, setup data path by changing ```base_path```

2. Run ```python run_train.py nelf_ft```

3. The intermediate results and model checkpoints are saved in ```[path_to_data_directory]/data_results/nelf_ft```

### Configs

The following config files can be found inside ```arg``` folder

* ```nelf_ft``` is our main model described in the paper

* ```ibr``` is our reimplementation of [IBRNet](https://ibrnet.github.io/)

* ```sipr``` is our reimplementation of [Single Image Portrait Relighting](https://cseweb.ucsd.edu//~viscomp/projects/SIG19PortraitRelighting/)


## Citation

```
@inproceedings {sun2021nelf,
    booktitle = {Eurographics Symposium on Rendering},
    title = {NeLF: Neural Light-transport Field for Portrait View Synthesis and Relighting},
    author = {Sun, Tiancheng and Lin, Kai-En and Bi, Sai and Xu, Zexiang and Ramamoorthi, Ravi},
    year = {2021},
}
```