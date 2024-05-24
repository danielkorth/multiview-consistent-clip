import torch
from typing import List
import numpy as np

from torchmetrics.functional.pairwise import pairwise_cosine_similarity


def loss_view_invariant_embedding_for_2_images(
        text_prompt_embedding: torch.tensor,
        distances: torch.tensor,
        predicted_img_embeddings: torch.tensor
    ) -> torch.tensor:
    """Compute loss for encouraging view invariant embeddings.
    
        text_prompt_embedding: torch.tensor (embedding size),
        distances: torch.tensor (2),
        predicted_img_embeddings: torch.tensor (2, embedding size)
        
    """
    loss = 0.0

    #compute loss for difference in distance to text prompt
    for i in range(2):
        loss  += \
        torch.max(
            torch.tensor(0.0).to(text_prompt_embedding.device), 
            torch.norm(text_prompt_embedding - predicted_img_embeddings[i]) - distances[i])


    # compute loss for distance between 2 predicted embeddings
    loss += \
        torch.norm(predicted_img_embeddings[0] - predicted_img_embeddings[1])
    
    return loss

def loss_view_invariant_embedding_for_n_images(
        text_prompt_embedding: torch.tensor,
        original_img_embeddings: List[torch.tensor],
        predicted_img_embeddings: List[torch.tensor]
    ) -> torch.tensor:
    """Compute loss for encouraging view invariant embeddings."""

    #TODO dont use lists
    #TODO take in distances in stead of original_img_embeddings
    num_img_embeddings = len(original_img_embeddings)

    loss = 0.0

    for i in range(num_img_embeddings):
        original_img_embedding = original_img_embeddings[i]
        predicted_img_embedding = predicted_img_embeddings[i]

        diff_in_dist_to_text_prompt = \
        torch.max(
            torch.tensor(0.0).to(text_prompt_embedding.device), 
            torch.norm(text_prompt_embedding - predicted_img_embedding) - torch.norm(text_prompt_embedding - original_img_embedding))

        loss += diff_in_dist_to_text_prompt

        neighbour_predicted_img_embedding = predicted_img_embeddings[(i+1) % num_img_embeddings]
        dist_to_neighbour_img_embedding = \
            torch.norm(predicted_img_embedding - neighbour_predicted_img_embedding)
        
        loss += dist_to_neighbour_img_embedding

    return loss

def loss_autoencoder_embedding(
        original_img_embeddings: torch.tensor,
        predicted_img_embeddings: torch.tensor
    ) -> torch.tensor:
    """Compute loss for autoencoder embeddings."""
    #TODO fix so that it works with size (batch size, data points size, embedding size)
    loss = torch.norm(original_img_embeddings - predicted_img_embeddings, dim=1)
    loss = torch.sum(loss)

    return loss

def _sum_diagonals_matrix(mat: torch.tensor, batch_size: int, datapoint_size: int): 

    # compute sim of al diagonals
    n,_ = mat.shape
    zero_mat = torch.zeros((n, n)) # Zero matrix used for padding
    mat_padded =  torch.cat((zero_mat, mat, zero_mat), 1) # pads the matrix on left and right
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

    return similarity_score_summed, dissimilarity_score_summed

def loss_contrastive(
        text_embeddings: torch.tensor,
        predicted_img_embeddings: torch.tensor
    ) -> torch.tensor:
    """
    text_embeddings: torch.tensor (batch size, embedding size), 
    img_embeddings: torch.tensor (batch size, data points size, embedding size)
    """
    batch_size, datapoint_size, embedding_size = predicted_img_embeddings.shape
    al_embeddings = torch.cat((text_embeddings.unsqueeze(1), predicted_img_embeddings), dim=1) #shape (batch size, data points size + 1, embedding size)
    permuted_embeddings = al_embeddings.permute(1, 0, 2) #shape (data points size + 1, batch size, embedding size)
    permuted_embeddings = permuted_embeddings.reshape(-1, embedding_size) #shape ((data points size + 1) * batch size, embedding size)

    sim = pairwise_cosine_similarity(permuted_embeddings)
    similarity_score_summed, dissimilarity_score_summed = _sum_diagonals_matrix(sim, batch_size, datapoint_size)

    sim_score_normalized = similarity_score_summed / (batch_size * (datapoint_size * (datapoint_size + 1) / 2))
    dissim_score_normalized = 0 if batch_size == 1 else dissimilarity_score_summed / (datapoint_size * (datapoint_size + 1) * batch_size * (batch_size - 1) / 2)

    weight_similarity = 1 #TODO make this a hyperparameter whenwe know how we want it to be
    loss = (1-weight_similarity)*dissim_score_normalized - weight_similarity*sim_score_normalized

    return loss, sim_score_normalized, dissim_score_normalized
