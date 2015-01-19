import nltk

# Simplest task - Tokenize a sentence
# The example sentence in English is interesting: 
#  1. o'clock - a contraction that nltk tokenizes as a single token 
#  2. didn't - tokenized as "did" "n't"
#  3. The final dot is marked as another token 
# TODO : Find an interesting example in Spanish

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
tokens = nltk.word_tokenize(sentence)
print tokens
