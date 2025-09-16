# Zero Shot Natural Language Temporal Video Grounding.
[📑 Paper](https://arxiv.org/abs/REPLACE) · [🌎 Project Page](https://soldelli.github.io/residualvit/)  · [💻 Training Code](https://github.com/Soldelli/residualvit)

Official PyTorch implementation of **ResidualViT for Efficient Temporally Dense Video Encoding**, accepted at **ICCV 2025 (highlight paper)**.  
This repository provides the testing code for NLTVG task.



## 🚀 Installation
This repository uses [PyTorchLighting](https://www.pytorchlightning.ai/). It also uses [Hydra](https://hydra.cc/docs/intro/) to manage runs configurations. We have facilitated a conda environment for quick setup. Assuming conda is installed, run:

```bash
conda env create -f environment.yml
conda activate sm
pip install --no-dependencies git+https://github.com/Soldelli/residualvit
export PYTHONPATH="$PYTHONPATH:$PWD"
```




## 🎯 Zero Shot Evaluation
To evaluate a model in particular dataset, follow the template below:

```bash
python -m aligner command=evaluate encoder=$MODEL data=$DATASET output_dir=$OUTPUT_DIR
```

### 📊 Supported datasets

| Dataset           |    Annotations    |    Videos     |  
|-------------------|------------------ |---------------|
| Charades-STA      | [Download](https://drive.google.com/file/d/1guZfHEsZwWCAm4NCttOzuufsLEA9l2jI/view?usp=sharing) | [Website](https://prior.allenai.org/projects/charades) |
|  ActivityNet-Captions   | [Download](https://drive.google.com/file/d/1ZI8GUuA7Tk5mblTlLVOgp2WTdXYA2lA9/view?usp=sharing) | [Website](http://activity-net.org/) |


### 🤖 Supported encoders 
* [OpenCLIP](https://github.com/mlfoundations/open_clip). This repository supports these available OpenCLIP  models.</br>
Available encoders: `openclip_vit_b_32`, `openclip_vit_b_16`, `openclip_vit_l_14`.
* [ResidualViT](https://github.com/Soldelli/residualvit). See [scripts](scripts/) for examples on how to use this encoder and visit the official ResidualViT codebase for the training code ([here](https://github.com/Soldelli/residualvit)).


## 📂 Repository Structure
```bash
zs-video-eval/
├── aligner/          # Core source code
├── configs/          # Config files for encoders and datasets
├── scripts/          # Training scripts
├── environment.yml   # Python dependencies
├── LICENSE.md        # Project license
├── README.md         # Project documentation
└── ...
```




## 💡 Citation
If you use this code or find it helpful in your research, please cite our papers:
```bibtex
@inproceedings{soldan2025residualvit,
  title={ResidualViT for Efficient Temporally Dense Video Encoding},
  author={Soldan, Mattia and Caba Heilbron, Fabian and Ghanem, Bernard and Sivic, Josef and Russell, Bryan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}

@article{castro2022fitclip,
  title={Fitclip: Refining large-scale pretrained image-text models for zero-shot video understanding tasks},
  author={Castro, Santiago and Heilbron, Fabian Caba},
  journal={arXiv preprint arXiv:2203.13371},
  year={2022}
}
```

## 🙏 Acknowledgements
This repository is built on top of [FitCLIP](https://github.com/bryant1410/fitclip), thanks to our collaborators and open-source community.

## 📜 License
This project is licensed under the [ADOBE RESEARCH LICENSE](LICENSE.md).
