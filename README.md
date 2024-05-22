#### Open Source LLM Inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ollama](https://github.com/ollama/ollama)
- [llamafile](https://github.com/Mozilla-Ocho/llamafile)
- [Huggingface]()
- [gpt4all](https://github.com/nomic-ai/gpt4all)
- [OpenLLM](https://github.com/bentoml/OpenLLM)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vllm](https://github.com/vllm-project/vllm)
  
#### LLM from scratch 

Pattern 1
  - Pretrained LLM
    - Prepare data
      - Use [datatrove](https://github.com/huggingface/datatrove/) lib and generate 15T dataset such as [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
      - Utilize available data
        - [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
        - For other datasets, refer LLM excel "Dataset" sheet
    - Training
      - Use [Nanotron](https://github.com/huggingface/nanotron) for training
    - Evaluation
      - Use [lighteval](https://github.com/huggingface/lighteval) for testing
      - Benchmarks
        - MMLU, [paper](https://arxiv.org/abs/2009.03300), [github](https://github.com/hendrycks/test), [standford](https://crfm.stanford.edu/2024/05/01/helm-mmlu.html)
        - For other benchmarks, refer LLM excel "[sheetname]"
  - SFT
  - RFHF
  - Instruct LLM
    - Datasets:
      - [CrystalCoderDatasets](https://huggingface.co/datasets/LLM360/CrystalCoderDatasets)

#### LLM deployment
- [bentoml](https://docs.bentoml.com/en/latest/use-cases/large-language-models/vllm.html)
- Managed services:
  - https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
  - https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2
- [MLFlow]()
