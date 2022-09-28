# A headline to be used as input 
headline = "Microsoft fails to hit profit expectations"

# Pre-process input phrase
input = tokenizer(headline, padding = True, truncation = True, return_tensors='pt')
# Run inference on the tokenized phrase
output = model(**input)

# Pass model output logits through a softmax layer.
sentim_scores = torch.nn.functional.softmax(output.logits, dim=-1)

