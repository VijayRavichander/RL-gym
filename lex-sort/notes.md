# Eval Split Performance

## deepseek-ai/DeepSeek-V3-0324 - DEEPINFRA INFRA PROVIDER

Dataset Size: 50
Accuracy: 0.54

# Base Model: Qwen2.5-0.5B

## Qwen/Qwen2.5-0.5B-Instruct 
Dataset Size: 50
Accuracy: 0.0

vijay-ravichander/Qwen2.5-0.5B-Lexo-Sort-SFT-v1 - Train Size: 500
Dataset Size: 50
Accuracy: 0.06

ONLY RL 
Killed the run. Flat Line!! Also confirms the notion that when your pre RL model is really bad like zero at your task then RL can't really help

SFT + RL - vijay-ravichander/Qwen2.5-0.5B-Lexo-Sort - 300 Steps
Dataset Size: 50
Accuracy: 0.08

we have got train-test leak
---------------------------------------------------------------------

# Base Model: Qwen3-1.7B 

## willcb/Qwen3-1.7B
Dataset Size: 50
Accuracy: 0.28

## Qwen3-1.7B-Lexo-Sort-SFT - Train Size: 500
Dataset Size: 50
Accuracy: 0.64

## Qwen3-1.7B-Lexo-Sort-GRPO
Dataset Size: 50
Accuracy: 0.72

## Qwen3-1.7B-Lexo-Sort-SFT-GRPO
Dataset Size: 50
Accuracy: 0.72
