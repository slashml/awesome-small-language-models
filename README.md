# Awesome Small Language Models

A curated list of awesome resources, tools, and projects related to small language models. This list focuses on modern, efficient language models designed for various applications, from research to production deployment.

## Table of Contents
- [Awesome Small Language Models](#awesome-small-language-models)
  - [Table of Contents](#table-of-contents)
  - [Some famous Small Language Models](#some-famous-small-language-models)
  - [Frameworks and Tools](#frameworks-and-tools)
  - [Fine-tuning Techniques](#fine-tuning-techniques)
    - [Fine-tuning Guide](#fine-tuning-guide)
  - [Hardware Requirements](#hardware-requirements)
  - [Inference Optimization](#inference-optimization)
  - [Applications and Use Cases](#applications-and-use-cases)
  - [Research Papers and Articles](#research-papers-and-articles)
  - [Tutorials and Guides](#tutorials-and-guides)
  - [Community Projects](#community-projects)
  - [Contributing](#contributing)
  - [License](#license)

## Some famous Small Language Models

- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) - A fine-tuned version of LLaMA, optimized for instruction following
- [Vicuna](https://github.com/lm-sys/FastChat) - An open-source chatbot trained by fine-tuning LLaMA
- [FLAN-T5 Small](https://huggingface.co/google/flan-t5-small) - A smaller version of the FLAN-T5 model
- [DistilGPT2](https://huggingface.co/distilgpt2) - A distilled version of GPT-2
- [BERT-Mini](https://huggingface.co/prajjwal1/bert-mini) - A smaller BERT model with 4 layers

## Frameworks and Tools

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0
- [Peft](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning (PEFT) methods
- [Periflow](https://github.com/periflow/periflow) - A framework for deploying large language models
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 8-bit CUDA functions for PyTorch
- [TensorFlow Lite](https://www.tensorflow.org/lite) - A set of tools to help developers run TensorFlow models on mobile, embedded, and IoT devices
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Cross-platform, high performance ML inferencing and training accelerator

## Fine-tuning Techniques

- [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685): Efficient fine-tuning method that significantly reduces the number of trainable parameters
- [QLoRA](https://arxiv.org/abs/2305.14314): Quantized Low-Rank Adaptation for even more efficient fine-tuning
- [P-tuning v2](https://arxiv.org/abs/2110.07602): Prompt tuning method for adapting pre-trained language models
- [Adapter Tuning](https://arxiv.org/abs/1902.00751): Adding small trainable modules to frozen pre-trained models

### Fine-tuning Guide

1. Choose a base model (e.g., FLAN-T5 Small, DistilGPT2)
2. Prepare your dataset for the specific task
3. Select a fine-tuning technique (e.g., LoRA, QLoRA)
4. Use Hugging Face's Transformers and Peft libraries for implementation
5. Train on your data, monitoring for overfitting
6. Evaluate the fine-tuned model on a test set
7. Optimize for inference (quantization, pruning, etc.)

## Hardware Requirements

RAM requirements vary based on model size and fine-tuning technique:

- Small models (e.g., BERT-Mini, DistilGPT2): 4-8 GB RAM
- Medium models (e.g., FLAN-T5 Small): 8-16 GB RAM
- Larger models with efficient fine-tuning (e.g., Alpaca with LoRA): 16-32 GB RAM

For training, GPU memory requirements are typically higher. Using techniques like LoRA or QLoRA can significantly reduce memory needs.

## Inference Optimization

- Quantization: Reducing model precision (e.g., INT8, FP16)
- Pruning: Removing unnecessary weights
- Knowledge Distillation: Training a smaller model to mimic a larger one
- Caching: Storing intermediate results for faster inference
- Frameworks for optimization:
  - [ONNX Runtime](https://github.com/microsoft/onnxruntime)
  - [TensorRT](https://developer.nvidia.com/tensorrt)
  - [OpenVINO](https://github.com/openvinotoolkit/openvino)

## Applications and Use Cases

- On-device natural language processing
- Chatbots and conversational AI
- Text summarization and generation
- Sentiment analysis
- Named Entity Recognition (NER)
- Question Answering systems

## Research Papers and Articles

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)

## Tutorials and Guides

- [Fine-tuning with LoRA using Hugging Face Transformers](https://huggingface.co/blog/lora)
- [Quantization for Transformers with ONNX Runtime](https://huggingface.co/blog/onnx-quantize-transformers)
- [Deploying Hugging Face Models on CPU with ONNX Runtime](https://huggingface.co/blog/onnx-runtime-inference)
- [Optimizing Inference with TensorFlow Lite](https://www.tensorflow.org/lite/performance/best_practices)

## Community Projects

- [Add your awesome community projects here!]


## Contributing

Your contributions are always welcome! Please read the contribution guidelines first.

## License

This awesome list is under the MIT License.