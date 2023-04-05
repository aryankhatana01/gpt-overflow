# gpt-overflow
A GPT-2 model finetuned on StackOverflow data to answer technical questions.
Maybe even a frontend to serve the model kinda like chatGPT w/o rlhf with
domain specific knowledge.

The Dataset: https://www.kaggle.com/datasets/stackoverflow/pythonquestions

In this repo I would be using the pretrained GPT-2 weights from
huggingface instead of training the whole model myself due to lack of GPUs :(

Steps to run the model:
1. Clone the repo
2. Install the requirements
3. Download the dataset from the link above
4. Unzip the dataset and place the `Questions.csv` and `Answers.csv` in the
   `data/` folder
5. Run the `2. Data Cleanup.ipynb` notebook to clean the data by removing
   unanswered questions from the dataset
