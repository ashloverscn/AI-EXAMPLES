import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

MODEL_PATH = "online_dictionary_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture Definition (Must match train.py)
class OnlineDictionaryModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx, hidden=None):
        x = self.embed(idx)
        out, hidden = self.rnn(x, hidden)
        logits = self.head(out)
        return logits, hidden

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint '{MODEL_PATH}' not found.")
        print("Please make sure you have successfully run 'train.py' first.")
        sys.exit(1)
        
    print(f"Loading dictionary model weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    chars = checkpoint['chars']
    vocab_size = len(chars)
    
    model = OnlineDictionaryModel(
        vocab_size=vocab_size,
        embed_dim=checkpoint['embed_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    eit_os = {i: ch for i, ch in enumerate(chars)}
    
    return model, stoi, eit_os

@torch.no_grad()
def generate(model, stoi, itos, start_string, max_new_chars=300, temperature=0.7):
    # Filter out characters that the model's vocabulary doesn't recognize
    valid_chars = [s for s in start_string if s in stoi]
    if not valid_chars:
        return "[Error: Starting characters/words are completely outside the trained dictionary vocabulary]"
    
    input_idx = torch.tensor([stoi[s] for s in valid_chars], dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated = list(valid_chars)
    
    # Warm up the hidden recurrent state with your prompt
    _, hidden = model(input_idx)
    current_idx = input_idx[:, -1:]
    
    # Generate definitions character-by-character
    for _ in range(max_new_chars):
        logits, hidden = model(current_idx, hidden)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        next_idx = torch.multinomial(probs, num_samples=1)
        next_char = itos[next_idx.item()]
        
        generated.append(next_char)
        current_idx = next_idx
        
    return "".join(generated)

def main():
    model, stoi, itos = load_trained_model()
    
    print("\n" + "="*50)
    print("=== AI Dictionary Interactive Prompt Ready ===")
    print("Type a word or term to generate its definition.")
    print("Example prompts: 'algorithm:', 'database:', 'compiler:'")
    print("Type 'exit' or 'quit' to close.")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("Dictionary Prompt > ")
            if prompt.strip().lower() in ["exit", "quit"]:
                print("Exiting dictionary interface. Goodbye!")
                break
            if not prompt.strip():
                continue
                
            print("\nGenerating definition...")
            output = generate(model, stoi, itos, start_string=prompt, max_new_chars=250)
            print("-" * 60)
            print(output)
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    main()