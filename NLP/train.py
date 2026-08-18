import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

# ==================== HYPERPARAMETERS ====================
MODEL_PATH = "online_dictionary_model.pth"
BATCH_SIZE = 64
BLOCK_SIZE = 128      
EMBED_DIM = 128       
HIDDEN_DIM = 256      
NUM_LAYERS = 2        
LEARNING_RATE = 2e-3
MAX_STEPS = 3000      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================================

def main():
    print("Fetching online English dictionary dataset from Hugging Face...")
    # Loading a public domain dictionary dataset directly from online sources
    dataset = load_dataset("npvinHnivqn/EnglishDictionary", split="train")
    
    print(f"Dataset downloaded successfully! Total dictionary entries: {len(dataset):,}")

    print("Compiling dictionary entries into a training stream...")
    text_corpus = []
    # Grab a subset or all records to form our definitions text block format: "word: definition\n"
    # Limiting to a subset first keeps memory nimble, or drop the slice [:10000] for all entries
    for item in dataset:
        word = item.get("word", "")
        definition = item.get("definition", "")
        if word and definition:
            text_corpus.append(f"{word}: {definition}")

    text = "\n".join(text_corpus)
    print(f"Corpus compilation complete. Total characters to train on: {len(text):,}")

    # Build character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]

    def get_batch():
        ix = torch.randint(len(train_data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([train_data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([train_data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

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

    print(f"Initializing model architecture on {DEVICE}...")
    model = OnlineDictionaryModel(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Starting training session on online data...")
    model.train()
    for step in range(MAX_STEPS):
        xb, yb = get_batch()
        logits, _ = model(xb)
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 500 == 0 or step == 0:
            print(f"Step [{step+1}/{MAX_STEPS}], Loss: {loss.item():.4f}")

    # Save model weights and character maps
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'chars': chars,
        'embed_dim': EMBED_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Training complete! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()