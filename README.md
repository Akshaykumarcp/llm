### LLM Business Use cases
- Customer engagement
  - Personalization & Customer segmentation
    - product/content rec based on behavior and preferences
  - Feedback analysis
    - what are top 5 customer complaints?
  - Virtual assistants
- Content creation
  - Creative writing: short stories, narratives, scripts, etc
  - Technical writing: doc, user manuals, simplify content, etc
  - Translation and localization
  - Article writing for blogs/social media
- Process automation and efficiency
  - Customer support augmentation and automated q&a
  - Automared customer response: Email, social media, product reviews
  - Sentiment analysis, prioritization
- Code generation and developer productivity
  - Code completion, boilerplate code generation
  - Error detection and debugging
  - Convert code between languages
  - Write code documentation
  - Automated testing
  - Natural language to code generation
  - Virtual code assistant for learning to code
- Summarization
- Classification
- QnA
- Named entity recognition


### Prompt engineering
- [Best practices](), [Techniques]()


### LLM Decoding strategies/methods
- [HF how to generate](https://huggingface.co/blog/how-to-generate), [code](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb)

### LLM components

- Tokenization
  - Option 1 - Transform text into word/token pieces
    - word into computational form
    - 2 steps
      - create vocab
      - Assign index
    - Cons
      - May miss few words
    - Pros
      - Intuitive
  - Option 2: Characters
    - vocab of around 100s, small
    - Cons
      - loose notion of word is
      - long sequence length post tokenization
  - Option 3: Byte pair encoding
  - Option 4: Sentence peice
  - Option 5: Word peice

- Word/token embeddings
  - Represent words with vectors
  - Option 1:
    - Count the frequency of the words in a document
    - Cons
      - SPARSITY
      - Sparse vectors lose meaningfull notio of similarity
  - Option 2:
    - give each word a vector represenation and use data to build embedding space
    - similar words, clustered together
    - example: word2vec
    - dimension sizes: 768, 1024, etc
    - Cons
      - Dense vector rep
- Tokenize + token embeddings is usually called encoding
- Language model
  - most likely next word
- Prompt

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

- Simplified training process

- Input - data
- Tokenize - encode text into numeric rep
- Token embeddings - Put words with similar meaning  close in vector space
- Transformer based model - Train
- Decoding - provide output that human understands (predict next word)
- Additional training: RLHF, etc


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
