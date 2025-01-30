from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def classificar_texto():
    textos = ["Ganhe dinheiro rápido", "Olá, como você está?", "Oferta imperdível!"]
    labels = ["spam", "normal", "spam"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(textos)
    modelo = MultinomialNB()
    modelo.fit(X, labels)

    entrada = input("Digite um texto para classificar: ")
    X_input = vectorizer.transform([entrada])
    predicao = modelo.predict(X_input)
    print(f"Classificação: {predicao[0]}")

classificar_texto()
