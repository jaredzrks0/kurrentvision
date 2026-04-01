# KurrentVision 

## Checkpoint 1

### The Problem and Evaluation Plan:

In 1937, my grandfather, Joseph Moses, immigrated to New York from Berlin, Germany, leaving most all of his family behind. In the years after, he remained in contact with his family via handwritten letters, all in German, and most in the Kurrent script. In the later years of his life, my mother and I, armed with limited German knowledge, have attempted to transcribe and translate these letters to gain a deeper understanding into his life, family, and life udner a growing Nazi regime back home in the late 1930s. While we have had success with most of these saved letters, there remain ~25 letters beyond our grasp, due to their complicated scripting and Kurrent font.

In this project, I aim via 2 methodologies to transcribe these remaining letters:

1. First, I will continue to fine tune the [TrOCR Kurrent-Model](https://huggingface.co/dh-unibe/trocr-kurrent). This is a Transformer based model, originally trained on general text recognition by Microsoft, and subsequently fine-tuned on a series of 18th century Kurrent texts by researchers at the University of Bern. I will fine tune their model with both a selection of previous letters we have succesfully transcribed, as well as additional synthetic data I will generate.

2. Second, I will train a model from scratch. If time permits, I will also aim to use an encoder/decoder transormer arcitecture, but a CNN may also be used if there is not enough time to train the transformer. Again, this will use both the transcribed letters and synthetic data, but also heavily rely on the available Senatsprotokolle documents from between 1795-1840 from the University of Greifswald.

Given the nature of the problem, there are multiple levels of success criteria:

1. At a minimum, I would like to be able to use the trained model to transcribe the untranscribed letters, either as it or with additional human interventions from the base text output. However, this is a very quantitative measure, so I also define two baselines.

2. Improvement upon the TrOCR model via fine-tuning. Given the aim is to transcribe personal letters, does finetuning on a subsection of already transcribed letters improve character error rate (CER) for other personal letters? CER on transcribed letters using TrOCR can be compared before and after fine-tuning.

3. Improvement over baseline [Transkribus](https://www.transkribus.org/kurrent-transcription). Trnaskribus is a software that offers transcription services in German and other languages, including in the Kurrent script. However, the software is paid to use, and their baseline German model, while not fully open sourced, appears to LSTM based. The website claims the model has an ~8% CER. While we do not direclty have their testing set, we aim to match or exceed their error rate on our data.


### Available Data
Data for this project comes from three major sources:

1. Handwritten letters my family has saved. This is a dataset of ~75 images, a selection of which are shown below. Of these 75, a subset of them already have ground truths, and the rest serve as our ultimate test for which we hope to transcribe. The ground truths letters too, serve as an interesting data source, as they are untouched by any pre-trained models as well, and offer a direct fine tuning ability for the ultimate test set.

| <img src="readme_images/GPA_IMG1.jpg" width="300"/> | <img src="readme_images/GPA_IMG2.jpg" width="300"/> | <img src="readme_images/GPA_IMG_3.jpg" width="300"/> |
|---|---|---|


2. Digitale Bibliothek Mecklenburg-Vorpommern maintains a digital archive of the [Senatsprotokolle](https://www.digitale-bibliothek-mv.de/viewer/toc/PPNUAG_0_1/3/), a series of handwritten notes from the academic senate at the University of Greifswald ranging from 1659-1937. A subset of these records have been prepared with text bounding info and transcriptions, allowing for me to use them as training data. I have collected the 16 available years with bounding info ranging between 1795 and 1840. These images can be downloaded year by year via the src/data_collection/senatsprotokolle_collection/image_scraping.py, and the bounding files can be downloaded directly from the digital archive. A few of the images can be seen below:

| <img src="readme_images/IMG1.jpg" width="300"/> | <img src="readme_images/IMG2.jpg" width="300"/> | <img src="readme_images/IMG3.jpg" width="300"/> |
|---|---|---|

3. Finally, I will use synthetic data for training + finetuning the proposed models. To generate this data, I am leveraging a python package called trdg which allows for the generation of text, and am creating said text with multiple Kurrent fonts downloaded from online. There are options in the package for image alterations like blur, background colors, rotations, etc. so the data generated should be valid and helpful for training purposes.Synthetic data can be generated from the src/data_collection/synthetic_text_generation/text_generator.py, where all the needed phrases are put into the data/synthetic_texts.txt file. While the parameters of this generation have not been determined yet, a few very simple images are shown below, simply showcasing the generation capability:

| <img src="readme_images/SYT_IMG_1.png" width="300"/> | <img src="readme_images/SYT_IMG_2.png" width="300"/> | <img src="readme_images/SYT_IMG_3.png" width="300"/> |
|---|---|---|

Beyond this, there are no major red flags in our data. This is classless, so there is no imbalance, and data is either synthetic or professionally generated, so there is data quality concerns either. 




### Evaluation Plan
With a data coming from multiple different sources, our data splitting with be a combination of inter and intra dataset splits. At a high level, I will create the standard train/val/test datset of all labeled data, and then we retain the final 'test' dataset of unlabeled data. The train/val/test datasets will consist of an approx even (by ratio) split of the real and synthetic data collected as described above, plus the pre-transcribed data from my grandfather. Once or model is trained, the final test set will consist of the untranscribed letters so see if we can acomplish the task at hand!

### Next steps
The immediate next step is the generation of the final dataset of synthetic data. Once this is complete and our final dataset is built, I will move onto the baseline models. This involves finetuning the downloaded TrOCR. Finally, once this is complete, I will move onto, assuming time and compute allows, to training my own model.


## Checkpoint 2
### Classical Baseline
For the shallow baseline, we use a simple linear model. Its inputs are a flattened RBG image, and its output is a vector of length max_len, where max_len is the length of the longest vector in the training data. This becomes in essense a character token prediction task, with padding tokens and EOS tokens included.

Because this model has no prior training, we train it on a combination dataset of the the collected Kurrent texts, as well as the synthetic data we generated, in approximately equal quantity to the kurrent texts. In all this was just under 10,000 text segments. That said, while this serves as a baseline of sorts, we do not expect it to be a very competitive one. This is because of the models simplicity, which will likely fail to fully capture the complex nature of writing as well as the spatial relationship inherient to a photo.

The model was was set up to train for 50 epochs, but it quickly became clear that, as expected, it was not learning to read the texts well. After fewer than 10 epochs, the loss plateaued, and minimal to no learning had occured. Plots outlining the learning process can be seen here for these 10 epochs:

![](models/basic_model/training_metrics.png)

As it can be seen, loss remains stagnant throughout the full training process, and character error rate (CER) is only ~10%. Because the simple linear baseline fails to learn anything meaningful about character recognition, we also move onto a second baseline - a pretrained KurrentOCR Model.

Qualitatively, we can also inspect a few example image texts and their prediction outputs. A subset are shown below:

```
true: 'Akademie quitiren zu wollen. Wäre'
  
pred: 'Auc e, gj idgc hgie lc  enloD  ou  a.ö,av cim.odsegn sme n ees ne- dnr-wteneeel-war<en'
---

true: 'mit den Professionen verbunden seyn,'

pred: 'Mac e,  j Eeac hHie     e  or  on  a. e  ccim.o.sein sme n eer nen dnr-wteneeel-war<en'
---

true: 'dies Cabinet aufzustellen seÿ, und endlich auch die'

pred: 'Aac eo  j Ee,c htie d   en or  on  a. e   ci¬l¬.sein sme noees nen dnr-wteneeel-war<en'
```

Again, we see that the model outpus are essentially nonsense. The model has no ability to distinguish texts from one another, which we can see in that all the outputs are the maximum length, and share similar characters with one another.

### KurrentOCR Model
For our deeper baseline, we use A pre-trained OCR model, finetuned by Jonas Widmer and Tobias Hode at the university of Bern. This model is a vision transformer originally trained in A paper called TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models, and fine tuned by Microsoft on the IAM handwritting dataset, which contains more than 110k handwriting samples.

By using a much more complex model, and in particular one that can learn the spacial relationships of both an image and human handwriting, we would expect the accuracy of this baseline to be much stronger than the linear baseline. Because our Kurrent texts share much overlap with the training data from the previous Kurrent fine tuning, we continue fine-tuning with the only 'unseen' data that we have - the synthetic data. 

Due to the complexity of the model, training cost is much higher than the linear model. As such, while we will complete a full training fun for the final project, for the purposes of this checkpoint, we run a limited training run with N epochs. Additionally, with the final goal being utilizing the trained model on our untranscribed letters from my grandfather, we retain this model for predicting those letters until the full training run can be completed. The training metrics for the deeper baseline are shown below:

![](models/kurrent_ocr/training_metrics.png)

We quickly see that the ViT model is much much better than the linear model, even with the very shortened training run. Average loss is much loer, and the CER is in the 2-3% range! This will be quite a difficult baseline to outdo, and even on it's own it would likely offer significant transcription abilities on our final test set, but it does set out a fun challenge to try and improve the model with a longer training run with synthetic data and eventually some already transcribed Kurrent letters from my grandfather. 

Finally, like before, we can also show a quantitave list of transcription attempts. They can be seen below:

```
true: 'Muth'
pred: 'Muth'

true: 'erneuert'
pred: 'erneuert'

true: 'Gebote'
pred: 'Gebote'

true: 'dem großen Betsaale des Zuchthauses ihr Gebet mit'
pred: 'dem großen Tatsaale des Zuchthauses ihr Gebet mit'

true: 'schwarze'
pred: 'schwarze'
```

Again, the ViT model shows not only great relative prediction improvement, but also great absolute prediction power. The model, albiet in a small sample was able to mostly accurately predict all 5 images's texts!