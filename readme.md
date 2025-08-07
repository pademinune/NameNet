# NameNet

NameNet is a project to experiment with various different machine learning models and architectures. Every model aims to classify any name as Male or Female based on its most common usage.

You can view model specific notes within the model folder named as: "{model_name}.txt".

2 Datasets have been used so far:  
Small (1,500 names): https://huggingface.co/datasets/aieng-lab/namexact  
Large (40,000 names): https://huggingface.co/datasets/aieng-lab/namextend

## Models

### V1 (582 parameters)

A DNN with 2 layers: 26 -> 20 -> 2

V1 is a dense network that uses a "characters in bag" approach. It takes a name as input and encodes it into a 26-dimensional vector containing the number of occuruences of each letter. Order of letters is not preserved.

### V2 (21,298 parameters)

A DNN with 16d embeddings and 2 layers: 160 -> 128 -> 2

### R1 (19,378 parameters)

An RNN with 16d embeddings, 128d hidden layer, and 128 -> 2 final dense layer

### V3 (12,882 parameters)

A DNN with 16d embeddings, 3 layers and fewer parameters: 160 -> 64 -> 32 -> 2

## Key Breakthroughs
- I JUST NOTICED THAT THE EXTENDED DATASET HAS WEIRD LABELS. For example, John was marked as both M and F in the 'gender' column. You need to view 'primary_gender' rather than 'gender'.
    - With this change, the models perform signiificantly better in training and loss is very low.

- Even after the change, v2 and v2.1 with the 30 neuron hidden layer still aren't able to fit the large dataset well (although better than before).

- v2.3 seemed to overfit the data (1000 epochs was a lot), and i trained v2.3.small on 200 epochs. It seems better at generalizing, but im not sure.

- At this point, v2 and r1 both fit the training data, but r1 seems to do better (adam, omar) despite having 10% fewer parameters. They both do get stuck on the same names outside of dataset.
    - I'm starting to think that I need a bigger dataset to do better on those names

## Takeaways
- the "characters in a bag" (v1) architecture does not preserve character order and thus is limited.
- The fixed length approach (v2) that takes the last 10 characters learns the importance of specific character locations
    - the model trained on the large dataset performs better, but not significantly (not 30x better)
    - v2 performs better than v1 on names like: (mira, amir), (omar, roma), (alan, lana)
- The first RNN (r1.0) architecture, despite being trained for 5 epochs on batch size of 1 and with ~0.3 loss, generalized suprisingly well
    - it got the anagrams correct, as well as omar. It detected tokyo as male
        - probably gap in extended dataset
- v3 is a deeper version of v2 (more layers) with half the number of parameters, but it does not do well in testing
- The RNN architecture seemed to generalize better than any of the dense networks even with less learnable parameters.
