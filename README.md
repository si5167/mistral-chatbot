# mistral-chatbot



! pip install accelerate peft bitsandbytes git+https://github.com/huggingface/transformers trl py7zr auto-gptq optimum
     
Collecting git+https://github.com/huggingface/transformers
  Cloning https://github.com/huggingface/transformers to /tmp/pip-req-build-2ud2xb3t
  Running command git clone --filter=blob:none --quiet https://github.com/huggingface/transformers /tmp/pip-req-build-2ud2xb3t
  Resolved https://github.com/huggingface/transformers to commit 897a826d830e8b1e03eb482b165b5d88a7a08d5f
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting accelerate
  Downloading accelerate-0.23.0-py3-none-any.whl (258 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 258.1/258.1 kB 3.8 MB/s eta 0:00:00
Collecting peft
  Downloading peft-0.5.0-py3-none-any.whl (85 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.6/85.6 kB 9.1 MB/s eta 0:00:00
Collecting bitsandbytes
  Downloading bitsandbytes-0.41.1-py3-none-any.whl (92.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 92.6/92.6 MB 9.1 MB/s eta 0:00:00
Requirement already satisfied: pip in /usr/local/lib/python3.10/dist-packages (23.1.2)
Collecting install
  Downloading install-1.3.5-py3-none-any.whl (3.2 kB)
Collecting trl
  Downloading trl-0.7.1-py3-none-any.whl (117 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 118.0/118.0 kB 15.4 MB/s eta 0:00:00
Collecting py7zr
  Downloading py7zr-0.20.6-py3-none-any.whl (66 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.7/66.7 kB 8.8 MB/s eta 0:00:00
Collecting auto-gptq
  Downloading auto_gptq-0.4.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 87.4 MB/s eta 0:00:00
Collecting optimum
  Downloading optimum-1.13.2.tar.gz (300 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 301.0/301.0 kB 37.3 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from accelerate) (1.23.5)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (23.2)
Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (5.9.5)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from accelerate) (6.0.1)
Requirement already satisfied: torch>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from accelerate) (2.0.1+cu118)
Collecting huggingface-hub (from accelerate)
  Downloading huggingface_hub-0.17.3-py3-none-any.whl (295 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 295.0/295.0 kB 28.7 MB/s eta 0:00:00
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from peft) (4.66.1)
Collecting safetensors (from peft)
  Downloading safetensors-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 57.3 MB/s eta 0:00:00
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers==4.35.0.dev0) (3.12.4)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.35.0.dev0) (2023.6.3)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers==4.35.0.dev0) (2.31.0)
Collecting tokenizers<0.15,>=0.14 (from transformers==4.35.0.dev0)
  Downloading tokenizers-0.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 99.3 MB/s eta 0:00:00
Collecting datasets (from trl)
  Downloading datasets-2.14.5-py3-none-any.whl (519 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 519.6/519.6 kB 49.5 MB/s eta 0:00:00
Collecting texttable (from py7zr)
  Downloading texttable-1.7.0-py2.py3-none-any.whl (10 kB)
Collecting pycryptodomex>=3.6.6 (from py7zr)
  Downloading pycryptodomex-3.19.0-cp35-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 91.4 MB/s eta 0:00:00
Collecting pyzstd>=0.14.4 (from py7zr)
  Downloading pyzstd-0.15.9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (412 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 412.3/412.3 kB 43.8 MB/s eta 0:00:00
Collecting pyppmd<1.1.0,>=0.18.1 (from py7zr)
  Downloading pyppmd-1.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (138 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 138.8/138.8 kB 17.0 MB/s eta 0:00:00
Collecting pybcj>=0.6.0 (from py7zr)
  Downloading pybcj-1.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.8/49.8 kB 3.5 MB/s eta 0:00:00
Collecting multivolumefile>=0.2.3 (from py7zr)
  Downloading multivolumefile-0.2.3-py3-none-any.whl (17 kB)
Collecting brotli>=1.0.9 (from py7zr)
  Downloading Brotli-1.1.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (3.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 84.2 MB/s eta 0:00:00
Collecting inflate64>=0.3.1 (from py7zr)
  Downloading inflate64-0.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (93 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 93.1/93.1 kB 11.8 MB/s eta 0:00:00
Collecting rouge (from auto-gptq)
  Downloading rouge-1.0.1-py3-none-any.whl (13 kB)
Collecting coloredlogs (from optimum)
  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 5.4 MB/s eta 0:00:00
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from optimum) (1.12)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (2023.6.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->accelerate) (4.5.0)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (3.1.2)
Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.10.0->accelerate) (2.0.0)
Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.10.0->accelerate) (3.27.6)
Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.10.0->accelerate) (17.0.2)
Collecting sentencepiece!=0.1.92,>=0.1.91 (from transformers==4.35.0.dev0)
  Downloading sentencepiece-0.1.99-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 80.5 MB/s eta 0:00:00
Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from transformers==4.35.0.dev0) (3.20.3)
Collecting humanfriendly>=9.1 (from coloredlogs->optimum)
  Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 11.7 MB/s eta 0:00:00
Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (9.0.0)
Collecting dill<0.3.8,>=0.3.0 (from datasets->trl)
  Downloading dill-0.3.7-py3-none-any.whl (115 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 115.3/115.3 kB 15.0 MB/s eta 0:00:00
Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (1.5.3)
Collecting xxhash (from datasets->trl)
  Downloading xxhash-3.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 194.1/194.1 kB 21.6 MB/s eta 0:00:00
Collecting multiprocess (from datasets->trl)
  Downloading multiprocess-0.70.15-py310-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.8/134.8 kB 16.9 MB/s eta 0:00:00
Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets->trl) (3.8.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.35.0.dev0) (3.3.0)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.35.0.dev0) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.35.0.dev0) (2.0.6)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.35.0.dev0) (2023.7.22)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from rouge->auto-gptq) (1.16.0)
Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->optimum) (1.3.0)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (23.1.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (6.0.4)
Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (4.0.3)
Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (1.9.2)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (1.4.0)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets->trl) (1.3.1)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.10.0->accelerate) (2.1.3)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets->trl) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets->trl) (2023.3.post1)
Building wheels for collected packages: transformers, optimum
  Building wheel for transformers (pyproject.toml) ... done
  Created wheel for transformers: filename=transformers-4.35.0.dev0-py3-none-any.whl size=7745571 sha256=914c731adae69cbfffa58fea7d990accba5172a838fe9830cf82e64ae701e356
  Stored in directory: /tmp/pip-ephem-wheel-cache-9qae8u3l/wheels/c0/14/d6/6c9a5582d2ac191ec0a483be151a4495fe1eb2a6706ca49f1b
  Building wheel for optimum (pyproject.toml) ... done
  Created wheel for optimum: filename=optimum-1.13.2-py3-none-any.whl size=395599 sha256=0f21447846579f34a5450f47c37d4cc020b397d6e90fc31fed6cd8b9ba4e1d88
  Stored in directory: /root/.cache/pip/wheels/6e/b7/2c/79405d98f0943373d8546daeae25a3d377f7659ca0cbe48699
Successfully built transformers optimum
Installing collected packages: texttable, sentencepiece, brotli, bitsandbytes, xxhash, safetensors, rouge, pyzstd, pyppmd, pycryptodomex, pybcj, multivolumefile, install, inflate64, humanfriendly, dill, py7zr, multiprocess, huggingface-hub, coloredlogs, tokenizers, transformers, datasets, accelerate, peft, trl, optimum, auto-gptq
Successfully installed accelerate-0.23.0 auto-gptq-0.4.2 bitsandbytes-0.41.1 brotli-1.1.0 coloredlogs-15.0.1 datasets-2.14.5 dill-0.3.7 huggingface-hub-0.17.3 humanfriendly-10.0 inflate64-0.3.1 install-1.3.5 multiprocess-0.70.15 multivolumefile-0.2.3 optimum-1.13.2 peft-0.5.0 py7zr-0.20.6 pybcj-1.0.1 pycryptodomex-3.19.0 pyppmd-1.0.0 pyzstd-0.15.9 rouge-1.0.1 safetensors-0.4.0 sentencepiece-0.1.99 texttable-1.7.0 tokenizers-0.14.1 transformers-4.35.0.dev0 trl-0.7.1 xxhash-3.4.1

from huggingface_hub import notebook_login
notebook_login()
     
VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments
from trl import SFTTrainer
import os

data = load_dataset("tatsu-lab/alpaca", split="train")
data_df = data.to_pandas()
data_df = data_df[:5000]
data_df["text"] = data_df[["input", "instruction", "output"]].apply(lambda x: "###Human: " + x["instruction"] + " " + x["input"] + " ###Assistant: "+ x["output"], axis=1)
data = Dataset.from_pandas(data_df)


tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ")
tokenizer.pad_token = tokenizer.eos_token


quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True, tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
                            "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
                            quantization_config=quantization_config_loading,
                            device_map="auto"
                        )


model.config.use_cache=False
model.config.pretraining_tp=1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)


training_arguments = TrainingArguments(
        output_dir="mistral-finetuned-alpaca",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=1,
        max_steps=250,
        fp16=True,
        push_to_hub=True
)


trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=512
)


trainer.train()
     
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
You passed `quantization_config` to `from_pretrained` but the model you're loading already has a `quantization_config` attribute and has already quantized weights. However, loading attributes (e.g. disable_exllama, use_cuda_fp16, max_input_length) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored.
Map:   0%|          | 0/5000 [00:00<?, ? examples/s]
/usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py:207: UserWarning: You passed a tokenizer with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to overflow issues when training a model in half-precision. You might consider adding `tokenizer.padding_side = 'right'` to your code.
  warnings.warn(
You're using a LlamaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
[250/250 22:51, Epoch 0/1]
Step	Training Loss
100	1.479000
200	1.248000
TrainOutput(global_step=250, training_loss=1.3423775177001953, metrics={'train_runtime': 1383.1959, 'train_samples_per_second': 1.446, 'train_steps_per_second': 0.181, 'total_flos': 309533448536064.0, 'train_loss': 1.3423775177001953, 'epoch': 0.4})

! cp -r /content/mistral-finetuned-alpaca /content/drive/MyDrive/
     
Inference

from google.colab import drive
drive.mount('/content/drive')
     
Mounted at /content/drive

from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("/content/mistral-finetuned-alpaca")

inputs = tokenizer("""###Human: Why mobile is bad for human? ###Assistant: """, return_tensors="pt").to("cuda")
     
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.

model = AutoPeftModelForCausalLM.from_pretrained(
    "/content/mistral-finetuned-alpaca",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda")
     

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)
     

import time
st_time = time.time()
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(time.time()-st_time)
     
###Human: Why mobile is bad for human? ###Assistant: 1. Mobile devices can be addictive and lead to a sedentary lifestyle.
2. Mobile devices can be harmful to mental health, causing anxiety and depression.
3. Mobile devices can be harmful to physical health, causing eye strain and headaches.
4. Mobile devices can be harmful to social relationships, causing people to become isolated and disconnected from others.
5. Mobile devices can be harmful to privacy, as they can track users' location and personal information.

12.296765804290771


     
