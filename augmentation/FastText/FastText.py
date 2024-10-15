from gensim.models.fasttext import FastText
import os

path = r"C:\\Users\\liamm\\OneDrive\\Documents\\GitRepositories\\Thesis\\Augmentation"

# 1) after using the UnzipFastText.py you should 
# have 2 files inside of the Augmentation folder: 
#   * crawl-300d-2M-subword.bin [<-- This is the model]
#   * crawl-300d-2M-subword.vec
model_path = os.path.join(path, "crawl-300d-2M-subword.bin")
model = FastText.load_fasttext_format(model_path)

similar_words = model.wv.most_similar('thou', topn=30)
for word, similarity in similar_words:
    print(f"{word}: {similarity}")