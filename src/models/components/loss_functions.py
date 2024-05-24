import torch
from typing import List
import numpy as np

from torchmetrics.functional.pairwise import pairwise_cosine_similarity

def loss_contrastive(
        text_embeddings: torch.tensor,
        predicted_img_embeddings: torch.tensor
    ) -> torch.tensor:
    """
    text_embeddings: torch.tensor (batch size, embedding size), 
    img_embeddings: torch.tensor (batch size, data points size, embedding size)
    """
    # rearrange embeddings so that we get (t_a, t_b, ..., img_a1, img_b1, ..., img_an, img_bn)
    batch_size, datapoint_size, embedding_size = predicted_img_embeddings.shape
    al_embeddings = torch.cat((text_embeddings.unsqueeze(1), predicted_img_embeddings), dim=1) #shape (batch size, data points size + 1, embedding size)
    permuted_embeddings = al_embeddings.permute(1, 0, 2) #shape (data points size + 1, batch size, embedding size)
    permuted_embeddings = permuted_embeddings.reshape(-1, embedding_size) #shape ((data points size + 1) * batch size, embedding size)

    # Compute similarity matrix
    sim = pairwise_cosine_similarity(permuted_embeddings)

    # compute sum of al diagonals in similarity matrix
    n,_ = sim.shape
    zero_mat = torch.zeros((n, n)) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, sim, zero_mat), 1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((n, 2*n), (3*n + 1, 1)) # Change the strides
    sum_diags = torch.sum(mat_strided, 0) # Sums the resulting matrix's columns

    #extract the sums corresponding to the diagonals with similarity scores and dissimilarity scores
    dim = batch_size + batch_size * datapoint_size #dim of sim matrix
    similarity_indexes = np.arange(batch_size, dim, batch_size) #index of diagonals with sim score
    sim_mask = torch.zeros(dim)
    sim_mask[similarity_indexes] = 1
    dissim_mask = 1 - sim_mask	

    shifted_sim_mask = torch.cat((torch.zeros(dim), sim_mask))
    shifted_dissim_mask = torch.cat((torch.zeros(dim), dissim_mask))

    similarity_scores = sum_diags * shifted_sim_mask
    similarity_score_summed = torch.sum(similarity_scores)

    dissimilarity_scores = sum_diags * shifted_dissim_mask
    dissimilarity_score_summed = torch.sum(dissimilarity_scores)

    # Normalize the scores wrt the number of datapoints and batch size
    sim_score_normalized = similarity_score_summed / (batch_size * (datapoint_size * (datapoint_size + 1) / 2))
    dissim_score_normalized = 0 if batch_size == 1 else dissimilarity_score_summed / (datapoint_size * (datapoint_size + 1) * batch_size * (batch_size - 1) / 2)

    weight_similarity = 0.5 #TODO make this a hyperparameter whenwe know how we want it to be
    loss = (1-weight_similarity)*dissim_score_normalized - weight_similarity*sim_score_normalized

    return loss, sim_score_normalized, dissim_score_normalized
