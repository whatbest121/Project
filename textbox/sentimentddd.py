from textblob import TextBlob

# Preparing an input sentence
sentence = '''The platform provides universal access to the world's best education, partnering with top universities and organizations to offer courses online.'''

analysisPol = TextBlob(sentence).polarity
analysisSub = TextBlob(sentence).subjectivity

print(analysisPol)
print(analysisSub)