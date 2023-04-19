# gpt-overflow
A GPT-2 model finetuned on StackOverflow data to answer technical questions.
Maybe even a frontend to serve the model kinda like chatGPT w/o rlhf with
domain specific knowledge.

The Dataset: https://www.kaggle.com/datasets/stackoverflow/pythonquestions

Finetuned Model Weights can be downloaded from here: https://www.kaggle.com/datasets/aryankhatana/gptoverflow-model-weights

(This model was trained by me on a GCP instance with an NVIDIA L4 and took around 8 days to train from 12 epochs. This model can still be trained for longer and give better results.)

Here is a sample output from the model:

```Input```: *I have installed tensorflow using pip. I am trying to run the following code:\n\nimport tensorflow as tf\n\nwith tf.device(\'/cpu:0\'):\n    tf.initialize_all_variables().run()\n\n\nBut I get the following error: ModuleNotFoundError: No module named 'tensorflow'*

```Output```: *I had the same problem. I had to install tensorflow using pip.\n\nsudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.9.0-py2-none-any.whl I hope this helps.*

The above link is made up since the model hasn't been finetuned with human feedback.

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
