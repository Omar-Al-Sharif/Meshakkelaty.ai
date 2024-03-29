# AI Arabic Tashkeel Engine - مِشَكِّلاتي.ai

<p>
  <a href="https://www.linkedin.com/posts/omar-al-sharif_last-month-our-ai-model-won-the-first-activity-7169065967623258112-VibU?utm_source=share&utm_medium=member_desktop">
  <img src="./Meshakkelaty.png" />
  </a>
</p>

## Overview
Welcome to مِشَكِّلاتي.ai ... An innovative Arabic text diacritization (Tashkeel) engine developed using advanced neural and statistical techniques. This project aims to accurately predict and add diacritics to Arabic text, enhancing readability, understandability, and Arabic text processing.
The مِشَكِّلاتي.ai model achieved [first-place on Kaggle](https://www.kaggle.com/competitions/cufe-cmp-credit-nlp-fall-2023/leaderboard), showcasing its exceptional performance 🥇

## Dual- Model Architecture
The مِشَكِّلاتي.ai diacritization system employs a dual-model architecture that consists of:
  1. A Neural Bidirectional Stacked Long Short-Term Memory (BiLSTM) model - that captures sequential dependencies and context information within the Arabic text - inspired by [this research paper](https://aclanthology.org/D19-5229/), but ***on steroids!***
  2. A Statistical Post-Processing model that operates on the output generated by the neural model to further refine the diacritization results, inspired by [this research paper](https://www.researchgate.net/publication/339041260_Arabic_Diacritic_Recovery_Using_a_Feature-Rich_biLSTM_Model)

https://github.com/Omar-Al-Sharif/Meshakkelaty.ai/assets/68480294/511d23a9-74d0-4fc0-a2e2-057b3ab79ee2

## Usage
To use مِشَكِّلاتي.ai, follow these steps:
1. Clone the repository
    - `git clone https://github.com/Omar-Al-Sharif/Meshakkelaty.ai.git`
2. Install the necessary dependencies:
    - `pip install -r Meshakkelaty.ai/requirements.txt `
3. Acquire your data and place them in `data` directory under the names `train.txt` and `val.txt`
4. Change the directory to scripts directory: 
    - `cd Meshakkelaty.ai/scripts`
5. Prepare your data by running the following command
    - `python tokenize_dataset.py`
6. Train the neural model on your data
    - `python train_neural_model.py`  
7. Train the statistical model on your data
    - `python train_statistical_model.py` 
8. Put your input text inside:
    - `../data/test_input.txt`
9. Diacritize the input text by running:
    - `python predict.py`

## Contributors
- [Omar Al Sharif](https://github.com/Omar-Al-Sharif)
- [Omar Atef](https://github.com/Yalab7/)
- [Omar Badr](https://github.com/Grintaking19)
- [Youssef Hany](https://github.com/youssefhassan01)
