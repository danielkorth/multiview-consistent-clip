import torch
from typing import List

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

def loss_contrastive(
        text_embeddings: torch.tensor,
        predicted_img_embeddings: torch.tensor
    ) -> torch.tensor:
    """
    text_embeddings: torch.tensor (batch size, embedding size), 
    img_embeddings: torch.tensor (batch size, data points size, embedding size)
    """
    batch_size, datapoint_size, embedding_size = predicted_img_embeddings.shape
    combined_embeddings = torch.cat((text_embeddings.unsqueeze(1), predicted_img_embeddings), dim=1)
    
    # TODO can concatenate to avoid looping over the batch.
    loss_similarity = 0
    loss_dissimilarity = 0

    for batch_idx in range(batch_size):
        sim =  pairwise_cosine_similarity(combined_embeddings[batch_idx])
        loss_similarity += torch.triu(sim, diagonal=1).sum()

    for batch_idx in range(batch_size):
        for nested_batch_idx in range (batch_idx + 1, batch_size):
            dissim = pairwise_cosine_similarity(combined_embeddings[batch_idx], combined_embeddings[nested_batch_idx])
            loss_dissimilarity += dissim[:, 1:].sum() 

    loss_sim_normalized = loss_similarity / (batch_size * (datapoint_size * (datapoint_size + 1) / 2))
    loss_dissim_normalized = 0 if batch_size == 1 else loss_dissimilarity / (datapoint_size * (datapoint_size + 1) * batch_size * (batch_size - 1) / 2)


    weight_similarity = 0.5 #TODO make this a hyperparameter whenwe know how we want it to be
    loss = (1-weight_similarity)*loss_dissim_normalized - weight_similarity*loss_sim_normalized

    return loss, loss_sim_normalized, loss_dissim_normalized