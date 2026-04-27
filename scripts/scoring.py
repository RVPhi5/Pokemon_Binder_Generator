import numpy as np 


def compute_page_embeddings(card_embeddings):
    """Compute page embedding p from card embeddings z1..zn"""

    if len(card_embeddings) == 0:
        raise ValueError("Page has 0 cards. Add cards to compute page embeddings")
    return np.mean(card_embeddings, axis=0)


def page_score (card_embeddings, model):
    """Computes overall page aesthetic score; averages compatbaility of each card with 
    page embedding """

    p = compute_page_embeddings(card_embeddings)

    scores = []
    for z in p: 
        score = model(p,z).item()
        scores.append(score)
    
    final_score = np.mean(scores)
    return final_score