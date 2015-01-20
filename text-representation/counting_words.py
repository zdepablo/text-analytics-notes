from nltk.book import *

# Some useful utilities from nltk to deal with text stats
# NLTK uses a Text data structure to support it 

# Number of tokens in text3
len(text3)

# Unique tokens sorted 
sorted(set(text3))

# Number of unique tokens (types) 
len(set(text3))

# Count the frequency of a token - the number of times a token occurs
text3.count("the")

# Frequency distribution 
fdist1 = FreqDist(text1)
fdist1.most_common(50)
fdist1.plot(50, cumulative = True)  # Plot the cumulative count for the 50 most frequent words 

# Find 'hapaxes' - hapax legomenoma are terms that appear just once
fdist1.hapaxes()

# Find long words - larger than 15 
V = set(text1) 
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

# Find words with more than 7 characters that appear at least 7 times 
fdist5 = FreqDist(text5) 
sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7)

# Bigrams 
bi = bigrams(text1) # bi is a generator
print list(bi)[0:50]

# Collocations - frequent bigrams that appear often - red wine vs maroon wine
text1.collocations()


