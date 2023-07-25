<a href="https://colab.research.google.com/github/[Your GitHub username]/gpt-neo-tabular-data/blob/main/gpt-neo-tabular-data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# GPT-NeoTab

"GPT-NeoTab is a powerful data generation system that employs the capabilities of GPT-Neo. The model's architecture is based on the transformer's self-attention mechanism, which has proven effective in understanding contextual relations between words in a text, to generate synthetic tabular data.

To accomplish its purpose, GPT-NeoTab works by encoding each row of your dataset into a sequential format, using each cell as an input token. The model then applies GPT-Neo's transformer-based machine learning algorithm to predict the probability distribution of possible next tokens in the sequence. By taking this approach, the generated synthetic data retains the overall statistical characteristics of your original dataset, such as mean, variance, correlation, and potential nonlinear dependencies between variables.

However, it is important to note that GPT-NeoTab doesn't just reproduce any specific row verbatim. It synthesizes completely new rows of data based on the learned patterns from the original dataset. This ensures that the privacy and confidentiality of the original data source are preserved, making GPT-NeoTab particularly valuable in sensitive domains where data privacy is crucial, such as healthcare, finance, and human resources.

In essence, GPT-NeoTab marries the prowess of the GPT-Neo's language understanding capabilities with the unique requirements of tabular data synthesis, providing a robust solution for generating high-quality, realistic, and privacy-preserving synthetic tabular data."
<p align="center">
<img src="https://github.com/[Your GitHub username]/gpt-neo-tabular-data/raw/main/img/dummy_image.png" style="width:75%"/>
</p>
<p align="center">
<strong>GPT-NeoTab: Generating Realistic Tabular Data using GPT-Neo</strong>
<br>
</p>
<br>

## Installation


```bash
Move the file LMsyn.py in your project's directory
```
## Training

```Python
from LM_syn import LMmodel

model = LMmodel()
model.fit(x_train,batch_size=8, epochs=50)
```
## Sampling

```Python
synthetic_data = model.sample(n_samples=999)
```
## Saving the weights of fine-tuned model

```Python
model.save(path)
```
## Loading the weights of fine-tuned model

```Python
from LM_syn import LMmodel

model = LMmodel()
model.set_metadata(x_train)
model.load_model(path+"/model.pt")
