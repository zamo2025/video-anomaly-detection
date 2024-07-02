# video-anomaly-detection

This repository was built with the goal of reproducing the results of paper "Learning Normal Dynamics in Videos with Meta Prototype Network" of the IEEE International Conference on Computer Vision and Pattern Recognition, written by Hui Lv, Chen Cehn, Cui Zhen, Chunysn Xu, and Jian Yang. The code in this repository was directly modeled off of the code for this paper. See their code here: [ktr-hubrt](https://github.com/ktr-hubrt/MPN).

Steps to run the code in Binghamton's openhpc server:

1. Set up a python virtual environment:
    ```bash
    python -m venv ./

2. Activate the environment:
    ```bash
    source ./bin/activate

3. Install the required pyhton packages:
    ```bash
    pip install -r requirements.txt

4. Load the cuda module:
    ```bash
    module load cuda

5. Copy the UCSD datasets to the data folder [UCSDped](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm).

6. Extract datasets:
    ```bash
    python3 extract_ped_datasets.py

7. Train a model using slurm:
    ```bash
    srun -n1 -N1 --partition=gpu python3 train_DPU.py

8. Test the model (this doesn't currently work):
    ```bash
    srun -n1 -N1 --partition=gpu python3 test_DPU.py

    ```
    @inproceedings{Lv2021MPN,
        author    = {Hui LV and
                Chen Chen and
                    Zhen Cui and
                    Chunyan Xu and
                    Yong Li and
                    Jian Yang},
        title     = {Learning Normal Dynamics in Videos with Meta Prototype Network},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year      = {2021}
    }