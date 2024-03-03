# CogVLM & CogAgent

## Setup

***Setting Up The RunPod Environment***

1. Basic Installations
```bash
apt-get update
apt-get install zip unzip
apt-get install git-lfs
git lfs install
```

2. Clone this repository and navigate to CogAgent folder
```bash
git clone https://github.com/abdur75648/CogAgent
cd CogAgent
```

3. Install the required packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
mkdir model
cd model
wget https://huggingface.co/THUDM/CogAgent/resolve/main/cogagent-vqa.zip?download=true -O cogagent-vqa.zip
unzip cogagent-vqa.zip
cd ../finetune_demo
bash finetune_cogagent_lora.sh
```

4. Run the demo
```bash
cd ../basic_demo
python web_demo.py --from_pretrained cogagent-vqa --version chat_old --bf16
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
