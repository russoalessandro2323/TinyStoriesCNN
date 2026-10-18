import src.dataset as d
from src.tokenizer import Tokenizer
import torch
import torch.nn as nn
from src.nnetwork import CNN
from src.LSTM import LSTM
from src.utils import save_checkpoint, load_checkpoint  
import os

def evaluate_model(model, dataloader, loss_function, t, device):
    """Computes validation loss over a subset of validation batches per epoch for speed."""
    model.eval()

    total_val_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            encoded_batch = t.batch_encode(batch)
            x = encoded_batch[:, :-1].to(device)
            y_true = encoded_batch[:, 1:].to(device)
            
            logits = model(x)
            loss = loss_function(logits, y_true)
                
            total_val_loss += loss.item()
            count += 1

    return total_val_loss / max(1, count)

def train_model():
    trn_raw_dataset, val_raw_dataset = d.get_dataset()

    dataloader = torch.utils.data.DataLoader(trn_raw_dataset, batch_size=32, shuffle=True) 
    val_dataloader = torch.utils.data.DataLoader(val_raw_dataset, batch_size=32, shuffle=True) 

    t = Tokenizer()
    t.load()

    vocab_size = len(t)
    n_epochs = 2
    embedding_dimension = 64
    n_res_blocks = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN(vocab_size, embedding_dimension, n_res_blocks).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    start_epoch = 0

    checkpoint_path = "./states/checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1 
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_train_loss = 0.0
    
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            encoded_batch = t.batch_encode(batch)  # (B, L)
            
            x = encoded_batch[:, :-1].to(device)
            y_true = encoded_batch[:, 1:].to(device)    

            logits = model(x)
            
            loss = loss_function(logits, y_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] | Step [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Finished | Avg Loss: {avg_train_loss:.4f} ---")

        avg_val_loss = evaluate_model(model, val_dataloader, loss_function, t, device)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss  

        checkpoint_state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "config": {
                "vocab_size": vocab_size,
                "embedding_dimension": embedding_dimension,
                "n_res_blocks": n_res_blocks,
            }
        }
        
        save_checkpoint(checkpoint_state, is_best=is_best)
        print()

    return 0


def train_lstm():
    trn_raw_dataset, val_raw_dataset = d.get_dataset()

    dataloader = torch.utils.data.DataLoader(trn_raw_dataset, batch_size=32, shuffle=True) 
    val_dataloader = torch.utils.data.DataLoader(val_raw_dataset, batch_size=32, shuffle=True) 

    t = Tokenizer()
    t.load()

    n_epochs = 1
    embedding_dimension = 64
    vocab_size = len(t)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTM(embedding_dimension, embedding_dimension, vocab_size).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    start_epoch = 0

    checkpoint_path = "./states/checkpoints/best_model_lstm.pt"
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1 
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_train_loss = 0.0
    
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            encoded_batch = t.batch_encode(batch).to(device)  # (B, L)
            
            x = encoded_batch[:, :-1]
            y_true = encoded_batch[:, 1:]

            logits = model(x)            
            loss = loss_function(logits, y_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] | Step [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Finished | Avg Loss: {avg_train_loss:.4f} ---")

        avg_val_loss = evaluate_model(model, val_dataloader, loss_function, t, device)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss  

        checkpoint_state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
            "config": {
                "embedding_dimension": embedding_dimension,
            }
        }
        
        save_checkpoint(checkpoint_state, is_best=is_best)
        print()

    return 0