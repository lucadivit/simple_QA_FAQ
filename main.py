from store_data import initialize_collection, do_a_question


initialize_collection()
print("Ciao! Sono il tuo assistente del sito Shoes Shop! Fammi una domanda!")
while True:
    sentence = input("> ")
    answer = do_a_question(sentence=sentence)
    print("<", answer)