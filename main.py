from src.tokenizer import Tokenizer
from src.nnetwork import CNN
import torch
from src.utils import load_checkpoint, top_k_neighbors, compute_cosine_similarity
from src.train import train_model, train_lstm
from src.LSTM import LSTM

def main():
  t = Tokenizer()
  t.load()

  vocab_size = len(t)
  embedding_dim = 64
  n_res_blocks = 7

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = CNN(vocab_size, embedding_dim, n_res_blocks).to(device)
  lstm = LSTM(embedding_dim, embedding_dim, vocab_size).to(device)
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
  optimizer_lstm = torch.optim.AdamW(lstm.parameters(), lr=0.001)

  load_checkpoint("./states/checkpoints/best_model.pt", model, optimizer)
  load_checkpoint("./states/checkpoints/best_model_lstm.pt", lstm, optimizer_lstm)

  prompts = ["One day I was sitting on a bench", "Once upon a time, there was a little girl named Lily", "The dog ran to the park and"]

  for prompt in prompts:
    print(" --- CNN ---  ")
    model.generate_story(prompt, t, t.L, device)
    print(" --- LSTM ---  ")
    lstm.generate_story(prompt, t, t.L, device)

  print("-" * 60)

  test_words = ["dad", "mum", "dog", "wizard", "river", "queen", "<UNK>", "a", "outside", "with", "knee", "trigonometria"]

  for word in test_words:
    top_k_neighbors(word, t, model, 10)

  print("-" * 60)

if __name__ == "__main__":
  main()
