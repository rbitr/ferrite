from sentence_transformers import SentenceTransformer
from time import time
import torch

#change as desired
#torch.set_num_threads(1)

if __name__ == "__main__":

    start = time()
    st_model =  SentenceTransformer("msmarco-distilbert-base-dot-prod-v3")
    end = time()

    print("load time:", end-start)

    with open("../sentence_ex.txt") as f:
        prompts = f.readlines()

    p = [p.strip() for p in prompts]
    print(len(p))
    #print(p)

    start = time()
    for p in prompts:
        #print(p)
        st_model.encode(p)

    end = time()
    print("inferenece time:", end-start)
