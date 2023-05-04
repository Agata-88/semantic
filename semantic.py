import spacy
nlp = spacy.load('en_core_web_md')

print ("------------------example 1--------------")
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

print ("------------------example 2--------------")
tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

print ("------------------example 3--------------")

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


'''
Task answers:

Example 1: cat is more similar to monkey than to banana, banana is
similar to monkey and cat vs apple has the least similarity. This shows that prgram
is able to detect that both cat and money are animals and that there is a link between
money and banana.

Example: when model is changed from en_core_web_md to en_core_web_sm detected 
similarity is much lower, this is due to the fact that sm method is not shipped with word vectors
and only uses context-sensitive tensors. There is a relevant statement printed on the terminal.

'''