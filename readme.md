# NameNet

NameNet is a project I've been working on to experiment with various different machine learning models and architectures. Every model aims to classify any name as Male or Female.
You can view model specific notes within the model folder named as: {model_name}.txt

2 Datasets have been used so far:
Small (1,500 names): https://huggingface.co/datasets/aieng-lab/namexact \n
Large (40,000 names): https://huggingface.co/datasets/aieng-lab/namextend

## Key Breakthroughs
- I JUST NOTICED THAT THE EXTENDED DATASET HAS WEIRD LABELS. For example, John was marked as both M and F in the 'gender' column. You need to view 'primary_gender' rather than 'gender'.
    With this change, the models perform signiificantly better in training and loss is very low.

- Even after the change, v2 and v2.1 with the 30 neuron hidden layer still aren't able to fit the large dataset well (although better than before).

- v2.3 seemed to overfit the data (1000 epochs was a lot), and i trained v2.3.small on 200 epochs. It seems better at generalizing, but im not sure.


## Takeaways
- the "characters in a bag" (v1) architecture does not preserve character order and thus is limited.
- The fixed length approach (v2) that takes the last 10 characters learns the importance of specific character locations
    - the model trained on the large dataset performs better, but not significantly (not 30x better)
    - v2 performs better than v1 on names like: (mira, amir), (omar, roma), (alan, lana)
- The first RNN (r1) architecture, despite being trained for 5 epochs on batch size of 1 and with ~0.3 loss, generalized suprisingly well
    - it got the anagrams correct, as well as omar. It detected tokyo as male
