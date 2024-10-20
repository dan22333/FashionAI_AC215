## Milestone 2 Fashion AI Project

[```
The files are empty placeholders only. You may adjust this template as appropriate for your project.
Never commit large data files,trained models, personal API Keys/secrets to GitHub
```]

#### Project Milestone 2 Organization

```
├── Readme.md
├── data # DO NOT UPLOAD DATA TO GITHUB, only .gitkeep to keep the directory or a really small sample
├── notebooks
│   └── eda.ipynb
├── references
├── reports
│   └── Statement of Work_Sample.pdf
└── src
    ├── datapipeline
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── dataloader.py
    │   ├── docker-shell.sh
    │   ├── preprocess_cv.py
    │   ├── preprocess_rag.py
    ├── docker-compose.yml
    └── models
        ├── Dockerfile
        ├── docker-shell.sh
        ├── infer_model.py
        ├── model_rag.py
        └── train_model.py
```

# AC215 - Milestone2 - Fashion AI App

**Team Members**
Yushu Qiu, Weiyue Li, Daniel Nurieli, Michelle Tan

**Group Name**
The Fashion AI Group

**Project**
Our goal is to create an AI-powered platform that aggregates fashion items from various brands, allowing users to quickly and easily find matching items without the hassle of endless browsing. Using our App, the users can put in a request such as "find me a classic dress for attending a summer wedding" and receive the clothing item that matches their request most closely.

### Milestone2 ###

For this milestone, we did the following: 
1. Scraped ~1,500 images and clothing items for men and women from the Farfetch website
2. Generated captions for the images using Gemini API
3. Finetuned the Fashion CLIP model using the images alongside their captions

In our submission file, we have the components for data scraping, caption generation, and Fashion CLIP finetuning. 

**Data scraping**

We gathered a dataset of 1,500 clothing items from the Farfetch website. We have stored it in a private Google Cloud Bucket. Note we limited the number of images used in this current training dataset because of budget and processing speed limitations related to scraping and caption generation. We plan to expand our dataset for future milestones. We used Apify API to automate the scraping process.

**Caption generation**

We used Gemini 1.5 Flash model to come up with captions for the images used for training. Here is the prompt we used: "For this image, come up with a caption that has 4 parts, and uses short phrases to answer each of the four categories below: - the style - the occasions that it’s worn in - material used - texture and patterns."

**Fashion CLIP finetuning**

[To add description]


-----
BELOW HAS NOT BEEN UPDATED

[**Data Pipeline Containers**
1. One container processes the 100GB dataset by resizing the images and storing them back to Google Cloud Storage (GCS).

	**Input:** Source and destination GCS locations, resizing parameters, and required secrets (provided via Docker).

	**Output:** Resized images stored in the specified GCS location.

2. Another container prepares data for the RAG model, including tasks such as chunking, embedding, and populating the vector database.

## Data Pipeline Overview

1. **`src/datapipeline/preprocess_cv.py`**
   This script handles preprocessing on our 100GB dataset. It reduces the image sizes to 128x128 (a parameter that can be changed later) to enable faster iteration during processing. The preprocessed dataset is now reduced to 10GB and stored on GCS.

2. **`src/datapipeline/preprocess_rag.py`**
   This script prepares the necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB).

3. **`src/datapipeline/Pipfile`**
   We used the following packages to help with preprocessing:
   - `special cheese package`

4. **`src/preprocessing/Dockerfile(s)`**
   Our Dockerfiles follow standard conventions, with the exception of some specific modifications described in the Dockerfile/described below.]


[## Running Dockerfile
Instructions for running the Dockerfile can be added here.
To run Dockerfile - `Instructions here`

**Models container**
- This container has scripts for model training, rag pipeline and inference
- Instructions for running the model container - `Instructions here`

**Notebooks/Reports**
This folder contains code that is not part of container - for e.g: Application mockup, EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations.]

----
You may adjust this template as appropriate for your project.


<section style="margin-top: 40px;">
  <!-- Your content here -->
</section>


#### ML model reports

<a href="https://github.com/weiyueli7/AC215_FashionAI/blob/michelle-test-branch/reports/W%26B%20Chart%2010_19_2024%2C%2010_13_28%20PM%20(3).svg">
    <img src="https://github.com/weiyueli7/AC215_FashionAI/blob/michelle-test-branch/reports/W%26B%20Chart%2010_19_2024%2C%2010_13_28%20PM%20(3).svg" alt="Interactive SVG" width="600" height="460" />
</a>


<a href="https://github.com/weiyueli7/AC215_FashionAI/blob/michelle-test-branch/reports/W%26B%20Chart%2010_19_2024%2C%2010_13_28%20PM%20(1).svg">
    <img src="https://github.com/weiyueli7/AC215_FashionAI/blob/michelle-test-branch/reports/W%26B%20Chart%2010_19_2024%2C%2010_13_28%20PM%20(1).svg" alt="Interactive SVG" width="600" height="460" />
</a>


<a href="https://github.com/weiyueli7/AC215_FashionAI/blob/michelle-test-branch/reports/W%26B%20Chart%2010_19_2024%2C%2010_13_28%20PM%20(2).svg">
    <img src="https://github.com/weiyueli7/AC215_FashionAI/blob/michelle-test-branch/reports/W%26B%20Chart%2010_19_2024%2C%2010_13_28%20PM%20(2).svg" alt="Interactive SVG" width="600" height="460" />
</a>
