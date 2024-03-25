# CogVLM & CogAgent

## Setup

***Setting Up The RunPod Environment*** (Template `2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`)

1. Basic Installations
```bash
apt-get update -y && apt-get install zip unzip vim -y && apt-get install git-lfs -y && git lfs install
```

2. Clone this repository and navigate to CogAgent folder
```bash
git clone https://github.com/abdur75648/CogAgent
cd CogAgent
```

3. Install packages one by one

3.1. Update pip and install the required packages
```bash
pip install --upgrade pip
pip install -r requirements.txt && python -m spacy download en_core_web_sm
unzip apex.zip && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

3.2. Install the required packages for the model
```bash
cd SwissArmyTransformer-0.4.11
pip install -e .
cd ../utils/models/GroundingDINO/ops
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
python setup.py build install
```

3.3. Download the model weights
```bash
cd ../../../../
cd model/cogagent-vqa/1/
wget https://huggingface.co/abdur75648/CogAgent-VQA/resolve/main/mp_rank_00_model_states.pt?download=true -O mp_rank_00_model_states.pt
cd ../../../
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```
<!-- Original Source of model - https://huggingface.co/THUDM/CogAgent -->

4. Run the finetune script
```bash
cd finetune_demo
bash finetune_cogagent_lora.sh
```

5. Run the demo
```bash
cd ../basic_demo
python web_demo.py --from_pretrained ../finetune_demo/checkpoints/finetune-cogagent-vqa-03-21-19-37/ --version chat_old --fp16 --use_lora
```

## Dataset
* Download '16k-Apollo-MLLM-data' folder inside current directory
* Run the following commands to extract the dataset
```bash
cd 16k-Apollo-MLLM-data
python3 get_16k_CogAgent_data.py
mkdir json
mv final_dataset.json json/apollo_ferret_noscale.json
```
* Replace the contents of `data/` folder with the contents of `16k-Apollo-MLLM-data` folder
bash
```bash
rm -rf data/
mv 16k-Apollo-MLLM-data/ data/
```


## Citation and references

```
@misc{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{hong2023cogagent,
      title={CogAgent: A Visual Language Model for GUI Agents}, 
      author={Wenyi Hong and Weihan Wang and Qingsong Lv and Jiazheng Xu and Wenmeng Yu and Junhui Ji and Yan Wang and Zihan Wang and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2312.08914},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
