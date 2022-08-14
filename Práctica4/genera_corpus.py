from Preprocesamiento.lematizador import lematizar
import os, pickle

def load_corpus(corpus):
	corpus_lematizado = []
	
	for linea in corpus:
		corpus_lematizado.append(lematizar(linea))

	return (corpus_lematizado)

if __name__ == "__main__":
	corpus = ['¡Pésimo! No gastes tu dinero ahí malas condiciones, deplorable. Definitivamente no gastes tu dinero ahí, mejor ve a gastarlo en dulces en la tienda de La catrina.',
          'La mejor vista de Guanajuato. Es un mirador precioso y con la mejor vista de la ciudad de Guanajuato. El monumento es impresionante. Frente al monumento (por la parte de atrás del Pípila) hay una serie de locales en donde venden artesanías... si te gusta algo de ahí, cómpralo. A mí me pasó que vi algo y no lo compré pensando que lo vería más tarde en otro lado y no fue así. Te recomiendo que llegues hasta ahí en taxi, son MUY económicos, porque como está en un lugar muy alto, es muy cansado llegar caminando, aunque no está lejos del centro. PEROOOO... bájate caminando por los mini callejones. ¡Es algo precioso!Te lleva directamente por un lado del Teatro Juárez.'
	]

	#Load lexicons
	if (os.path.exists('corpus_lematizado.pkl')):
		corpus_file = open ('corpus_lematizado.pkl','rb')
		corpus_lematizado = pickle.load(corpus_file)
	else:
		print ('no existe...')
		corpus_lematizado = load_corpus(corpus)
		corpus_file = open ('corpus_lematizado.pkl','wb')
		pickle.dump(corpus_lematizado, corpus_file)
		corpus_file.close()

	print (corpus_lematizado)	

