"""
red.py
~~~~~~~~~~
Es la definición de nuestra Red Neuronal básica.
Se implementa el algoritmo de aprendizaje de Gradiente Descendente Estocástica (GDE) 
(Stochastic Gradient Descent o SGD) [REF1] para la fase de aprendizaje. 
Las gradientes/pendientes se calculan mediante el algoritmo de Propagación 
Hacia Atrás (Backpropagation) [REF2].

IMPORTANTE: 

El código está implementado para ser entendible y puramente académico.
No está optimizado ni es su propósito.
Las referencias para lectura y aprendizaje están al final del fichero. Disfrútalo!

---

TRADUCTOR: Ibon Reinoso Isasi, http://ibonreinoso.eu/
CONTACTO: ibonreinoso@opendeusto.es, LinkedIn: https://www.linkedin.com/in/ibon-reinoso-isasi/ 
Última fecha rev: 19/12/2019, por Ibon Reinoso Isasi

---

Si tienes alguna duda, sugerencia de corrección o incluso quieres colaborar/aportar algo, no dudes en escribirme un correo!


"""


#### Librerias

# Librerias estándar
import random

# Librerias de terceros
import numpy as np

"""
Toda Red Neuronal se puede describir por ser compuesto por:
* Número de Capas
* Tamaño para cada capa, el número de neuronas 
* Sesgo 
* Peso
"""
class Red(object):

    def __init__(self, tamano):

        """
		La lista ``tamano`` contiene el número de neuronas para cada capa
		de la red. Por ejemplo, si la lista es [2,3,1] la red diseñada 
		sería una red de tres capas donde en la primera capa tendría 2 neuronas,
		en la segunda capa 3 neuronas y por último, la tercera capa, con una neurona. 
		Los sesgos y pesos asignados a la red se inicializan de manera aleatoria,
		utilizando una distribución Gausiana/Normal (de media 0 y varianza 1) [REF3].
		La primera capa se asume que son las de entrada y por convención, no utilizamos
		sesgos para estas neuronas (porque no se usan).
 
        """

        self.num_capas = len(tamano)
        self.tamano = tamano
        self.sesgo = [np.random.randn(y, 1) for y in tamano[1:]]
        self.pesos = [np.random.randn(y, x)
                        for x, y in zip(tamano[:-1], tamano[1:])]

   
    """
    Las funcionalidades que vamos a definir para nuestra Red Neuronal son las siguientes:
    * Función alimentacionHaciaDelante
    * Función GDE (gradienteDescendienteEstocástica)
    * Función actualizacionMiniLote para actualizar de manera parcial la Red Neuronal basándonos en 
    la función propagaciónHaciaAtrás. Lo hacemos así para que sea más rápido, como se suele decir: divide y vencerás!
    * Función propagacionHaciaAtras
    * Función evaluacion
    * Función costeDerivada
    Otras funciones...
    * Función sigmoide
    * Función derivadaSigmoide
    """

	def alimentacionHaciaDelante(self, a):
	"""Devuelve el resultado de nuestra Red para una entrada ``a``."""
	        for b, w in zip(self.sesgo, self.pesos):
	            a = sigmoide(np.dot(w, a)+b)
	        return a


	def GDE(self, datos_entrenamiento, ciclos, tamano_mini_lote, eta,
            datos_test=None):

        """Entrena la Red utilizando el mini-lote junto a el mecanismo de 
        Gradiente Descendiente Estocástica. 
        - Los ``datos_entrenamiento`` son pares ``(x, y)``. X son las entradas, mientras que Y 
        las salidas esperadas.  
		- Si se utiliza la funcionalidad con ``datos_test`` se muestra el progreso para cada ciclo
		(suele ser útil para ver el progreso pero ralentiza el proceso muchísimo).
        
        # El valor ``eta`` es el salto de avance para aplicar en GDE. Básicamente, es el tamaño 
        de salto (límite para saber cuánto quieres avanzar mientras aprende la Red). 
        Se hace para que no salte mucho más de la cuenta mientras la red está siendo entrenada [REF4]  
            
        """

        if datos_test: n_test = len(datos_test)
        n = len(datos_entrenamiento)
        for j in xrange(ciclos):
            random.shuffle(datos_entrenamiento)
            mini_lotes = [
                datos_entrenamiento[k:k+lote_size]
                for k in xrange(0, n, lote_size)]
            for lote in mini_lotes:
                self.actualizacionMiniLote(lote, eta)  
            if datos_test:
                print "Ciclo {0}: {1} / {2}".format(
                    j, self.evaluacion(datos_test), n_test)
            else:
                print "Ciclo {0} completado".format(j)

	
    def actualizacionMiniLote(self, mini_lote, eta):
            """Actualizamos los pesos y sesgo para nuestra Red mediante la téncica GDE para nuestro pequeño lote.
            ``mini_lote``se compone de pares ``(x, y)`` (igual que antes, salida real y esperada respectivamente)
            ``eta`` es la tasa de aprendizaje."""

            nabla_b = [np.zeros(b.shape) for b in self.sesgo]
            nabla_w = [np.zeros(w.shape) for w in self.pesos]
            for x, y in mini_lote:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.pesos = [w-(eta/len(mini_lote))*nw
                            for w, nw in zip(self.pesos, nabla_w)]
            self.sesgo = [b-(eta/len(mini_lote))*nb
                           for b, nb in zip(self.sesgo, nabla_b)]


    def propagacionHaciaAtras(self, x, y):
        """La función devuelve pares ``(nabla_b, nabla_w)``, funciones de coste C_x.
        Las variables ``nabla_b`` y ``nabla_w`` son listas vectorizadas con la librería
        Numpy. Tienen la misma estructura que ``self.sesgo`` y la variable ``self.pesos``.
        """

        nabla_b = [np.zeros(b.shape) for b in self.sesgo]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]
        
        # FASE 1: Alimentación Hacia Delante
        activation = x
        activations = [x] # Lista para almacenar las activaciones para cada capa
        zs = [] # Lista para almacenar los vectores z, para cada capa
        for b, w in zip(self.sesgo, self.pesos):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoide(z)
            activations.append(activation)

        # FASE 2: Porpagación Hacia Atrás
        delta = self.CosteDerivada(activations[-1], y) * \
            derivadaSigmoide(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # l = 1 significa la última capa de las neuronas. Empezamos desde el final y vamos volviendo al inicio, capa a capa.
        # Cuando l = 2 entonces será la penútima capa; l = 3 significaría la tercera capa, empezando por el final. 
        # Aprobechamos Python ya que permite utilizar listas inversas.

        for l in xrange(2, self.num_capas):
            z = zs[-l]
            sp = derivadaSigmoide(z)
            delta = np.dot(self.pesos[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluacion(self, datos_test):
        """Devuelve la suma de aciertos (es decir, cuantos X han sido Y).
        Esto lo que nos indica es cuantos aciertos totales hay frente a las salidas calculadas. """
        test_results = [(np.argmax(self.alimentacionHaciaDelante(x)), y)
                        for (x, y) in datos_test]
        return sum(int(x == y) for (x, y) in test_results)

    def costeDerivada(self, output_activations, y):
        """Devuelve la derivada parcial C_x para las salidas."""
        return (output_activations-y)

    #### Otras funcionalidades...
    def sigmoide(z):
        """Función sigmoide. [REF5] """
        return 1.0/(1.0+np.exp(-z))

    def derivadaSigmoide(z):
        """Derivada de la función sigmoide."""
        return sigmoide(z)*(1-sigmoide(z))


"""
Gracias por leerlo!
Quieres saber más? Aquí tienes las referencias, espero que sean útiles para ti :)
[REF1] Gradiente Descendente Estocástica (Stochastic Gradient Descent o SGD) * https://en.wikipedia.org/wiki/Stochastic_gradient_descent
[REF2] Propagación Hacia Atrás (Backpropagation) * https://en.wikipedia.org/wiki/Backpropagation
[REF3] Distribución Gausiana * https://en.wikipedia.org/wiki/Normal_distribution
[REF4] Gradiente Descendente * https://en.wikipedia.org/wiki/Gradient_descent (Dibujo, la distancia máxima para cada X)
[REF5] Función Sigmoide * https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide
--
# Contenido Original: Neural Networks and Deep Learning, de Michael Nielsen
# http://neuralnetworksanddeeplearning.com/
"""