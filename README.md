Occlusion Detection
===

![occlusion_results_compressed](https://github.com/EnzeXu/Occlusion_Detection/assets/90367338/5f95924b-a4c3-4f36-8f22-604f041decc4)

===

# Contents

* [1 Introduction](#1-introduction)
* [2 Citation](#2-citation)
* [3 Structure of the Repository](#3-structure-of-the-repository)
* [4 Getting Started](#4-getting-started)
  * [4.1 Preparations](#41-preparations)
  * [4.2 Install Packages](#42-install-packages)
* [5 Questions](#5-questions)



# 1. Introduction
Invariant Physics. [TODO]

# 2. Citation

If you use our code or datasets from `https://github.com/EnzeXu/Occlusion_Detection` for academic research, please cite the following paper:

Paper BibTeX:

```
@article{xxx2023xxxxxx,
  title        = {xxxxx},
  author       = {xxxxx},
  journal      = {arXiv preprint arXiv:xxxx.xxxx},
  year         = {2023}
}
```



# 3. Structure of the Repository

[TODO]

[//]: # ()
[//]: # (```)

[//]: # (SB-FNN)

[//]: # (┌── SBFNN/)

[//]: # (├────── models/)

[//]: # (├────────── _template.py)

[//]: # (├────────── model_rep3.py)

[//]: # (├────────── model_rep6.py)

[//]: # (├────────── model_sir.py)

[//]: # (├────────── model_asir.py)

[//]: # (├────────── model_turing1d.py)

[//]: # (├────────── model_turing2d.py)

[//]: # (├────── utils/)

[//]: # (├────────── __init__.py)

[//]: # (├────────── _run.py)

[//]: # (├────────── _utils.py)

[//]: # (├── LICENSE)

[//]: # (├── README.md)

[//]: # (├── requirements.txt)

[//]: # (└── run.py)

[//]: # (```)

[//]: # ()
[//]: # (- `ChemGNN/models/`: folder contains the model scripts)

[//]: # (- `ChemGNN/utils/`: folder contains the utility scripts)

[//]: # (- `LICENSE`: license file)

[//]: # (- `README.md`: readme file)

[//]: # (- `requirements.txt`: main dependent packages &#40;please follow section 3.1 to install all dependent packages&#41;)

[//]: # (- `run.py`: training script)



# 4. Getting Started

This project is developed using Python 3.9+ and is compatible with macOS, Linux, and Windows operating systems.

## 4.1 Preparations

(1) Clone the repository to your workspace.

```shell
~ $ git clone https://github.com/EnzeXu/Occlusion_Detection.git
```

(2) Navigate into the repository.
```shell
~ $ cd Occlusion_Detection
~/Occlusion_Detection $
```

(3) Create a new virtual environment and activate it. In this case we use Virtualenv environment (Here we assume you have installed the `virtualenv` package using you source python script), you can use other virtual environments instead (like conda).

For macOS or Linux operating systems:
```shell
~/Occlusion_Detection $ python -m venv ./venv/
~/Occlusion_Detection $ source venv/bin/activate
(venv) ~/Occlusion_Detection $ 
```

For Windows operating systems:

```shell
~/Occlusion_Detection $ python -m venv ./venv/
~/Occlusion_Detection $ .\venv\Scripts\activate
(venv) ~/Occlusion_Detection $ 
```

You can use the command deactivate to exit the virtual environment at any time.

## 4.2 Install Packages

```shell
(venv) ~/Invariant_Physics $ pip install -r requirements.txt
```

[//]: # (## 4.3 Build Datasets)

[//]: # ()
[//]: # (&#40;1&#41; Create the ODE datasets you want to run. Please follow the following instructions or use command `python make_datasets.py --help` to see all possible arguments.)

[//]: # ()
[//]: # (Example:)

[//]: # ()
[//]: # (```shell)

[//]: # (&#40;venv&#41; ~/Occlusion_Detection $ python make_datasets.py --params_strategy default --seed 0 --ode_name Lotka_Volterra --save_figure 1 --noise_ratio 0.05)

[//]: # (```)

[//]: # ()
[//]: # (You can run the script from the command line with various options. Here's a breakdown of the available command-line arguments:)

[//]: # ()
[//]: # (| Argument            | Description                                                                                     |)

[//]: # (|---------------------|-------------------------------------------------------------------------------------------------|)

[//]: # (| `--[TODO]`          | Specify the xxxxxx                                                                              |)

[//]: # (| `--[TODO]`          | Specify the xxxxxx                                                                              |)
[//]: # (&#40;2&#41; Run Training. [TODO])


[//]: # (You can combine these arguments according to your requirements to run the script with the desired settings. E.g.,)

[//]: # ()
[//]: # (```shell)

[//]: # (&#40;venv&#41; ~/SB-FNN $ python run.py --seed 999 --test 1 --cyclic 1)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (&#40;3&#41; Collect the auto-generated training results in `saves/figure/` and `saves/train/`.)

[//]: # (```shell)

[//]: # (&#40;venv&#41; ~/SB-FNN $ ls saves/train/MODEL_NAME_YYYYMMDD_HHMMSS_f/)

[//]: # (&#40;venv&#41; ~/SB-FNN $ ls saves/figure/MODEL_NAME_YYYYMMDD_HHMMSS_f/)

[//]: # (```)



# 5. Questions

If you have any questions, please contact xezpku@gmail.com.


