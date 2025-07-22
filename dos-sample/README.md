# Overthink Sample Generation and Reasoning Evaluation Toolkit

Our work provides two core scripts for generating "overthink" case samples and evaluating reasoning accuracy using large language model APIs.

## 🛠 Core Scripts

### 1. `sample.py`
- **Function**: Generates diverse overthink case samples with different trigger strength through API calls
- **Supported APIs**:
  - ✅ DeepSeek API (default)
  - 🔄 QwQ API
  - 🔄 OpenAI API
- **Output**: `sft-sample` (contains generated case texts and metadata)

### 2. `judge.py`
- **Function**: Evaluates reasoning accuracy
- **Metrics**:
  - Exact Match (EM)

## 🔌 API Configuration
Switch between providers using environment variables.

## ⚠️ Important Notice

Due to equipment failure, the following scripts are currently temporarily unavailable and will be added to the repository soon:

- sample.py (to be added shortly)
- judge.py (to be added shortly)
