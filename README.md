# IR3D-Bench: Evaluating Vision-Language Model Scene Understanding as Agentic Inverse Rendering





Abstract: *Vision-language models (VLMs) excel at descriptive tasks, but whether they truly understand scenes from visual observations remains uncertain. We introduce IR3D-Bench, a benchmark challenging VLMs to demonstrate understanding through active creation rather than passive recognition. Grounded in the analysis-by-synthesis paradigm, IR3D-Bench tasks Vision-Language Agents (VLAs) with actively using programming and rendering tools to recreate the underlying 3D structure of an input image, achieving agentic inverse rendering through tool use. This ''understanding-by-creating'' approach probes the tool-using generative capacity of VLAs, moving beyond the descriptive or conversational capacity measured by traditional scene understanding benchmarks. We provide a comprehensive suite of metrics to evaluate geometric accuracy, spatial relations, appearance attributes, and overall plausibility. Initial experiments on agentic inverse rendering powered by various state-of-the-art VLMs highlight current limitations, particularly in visual precision rather than basic tool usage. IR3D-Bench, including data and evaluation protocols, is released to facilitate systematic study and development of tool-using VLAs towards genuine scene understanding by creating.*




# Environment setup
(1) Create Environment:
```shell
conda create --name ir3d python=3.10
conda activate ir3d
```

(2) First install [vllm](https://github.com/vllm-project/vllm)
```
pip install vllm
```

(3) Install Blender [on linux](https://docs.blender.org/manual/en/latest/getting_started/installing/linux.html)
```shell
snap install blender --classic
```
(4) Install [SAM](https://github.com/facebookresearch/segment-anything)
```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

# Dataset setup
Download our processed data: [IR3D-bench-data](https://huggingface.co/datasets/Piang/IR3D-bench).

## Inverse Rendering
### Task prompt
Prompt for inverse rendering and gpt4o score is in `prompts/gpt4o_as_evaluator.txt` and `prompts/vlm_estimate_params.txt`
### Latest Proprietary Models
Modified the `model-name` as defined in `main_vllm.py` to use the required model.
```shell
python main_vllm.py --model-type "model-name"
```
### Open-source Models
Modified the `model-name` as you needed, such as "gpt-4o", "grok-3", etc.
```shell
python main_api.py \ 
    --image_dir /path/to/images \ 
    --result_dir /output/path \ 
    --prompt_path prompts/vlm_estimate_params.txt \ 
    --model_name "model-name"
```
## Eval
```shell
bash cal_metric.sh "/output/path" "/path/to/images" "GPI_ID"
```


## Acknowledgement
Thanks to the following fantastic repos:
- [SAM](https://github.com/facebookresearch/segment-anything)
- [vllm](https://github.com/vllm-project/vllm)
- [Clever dataset](https://github.com/facebookresearch/clevr-dataset-gen)
- [Blender](https://www.blender.org/)