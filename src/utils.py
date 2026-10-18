import os
import torch
import torch.nn.functional as F

def save_checkpoint(state, is_best=False, checkpoint_dir="./states/checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    last_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    torch.save(state, last_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_path)
        print(f" -> Best model saved with val loss: {state['val_loss']:.4f}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    start_epoch = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    
    print(f"Loaded checkpoint from '{checkpoint_path}' (Epoch {start_epoch})")
    return start_epoch, val_loss

def compute_cosine_similarity(word_1, word_2, t, model):

    """
        Returns a similarity value ∈ [0, 1] computed using cosine similarity
    """

    if word_1 not in t.word2idx or word_2 not in t.word2idx:
        print(f"One/both words ('{word_1}', '{word_2}') not found in vocabulary.")
        return None

    embedding_matrix = model.embedding.embedding_matrix.detach()                   
    embedding_matrix = embedding_matrix - embedding_matrix.mean(0, keepdim=True)                               
    normalized_embedding_matrix = F.normalize(embedding_matrix, dim=1)
        
    v_word_1 = normalized_embedding_matrix[t.word2idx[word_1]]
    v_word_2 = normalized_embedding_matrix[t.word2idx[word_2]]

    return torch.dot(v_word_1, v_word_2).item()

def top_k_neighbors(word, t, model, k=8):

    """
        Returns a k-vector composed by the most similar words with respect to the given word
    """

    embedding_matrix = model.embedding.embedding_matrix.detach()                   
    embedding_matrix = embedding_matrix - embedding_matrix.mean(0, keepdim=True)                               
    normalized_embedding_matrix = F.normalize(embedding_matrix, dim=1)

    try:
        
        word_id = t.word2idx[word]
        top = (normalized_embedding_matrix @ normalized_embedding_matrix[word_id]).topk(k + 1).indices.tolist() 
        print(word, "->", [t.idx2word[i] for i in top if i != word_id][:k])

    except:
        top = (normalized_embedding_matrix @ normalized_embedding_matrix[2]).topk(k + 1).indices.tolist()  # <UNK>
        print("<UNK> ->", [t.idx2word[i] for i in top if i != 2][:k])