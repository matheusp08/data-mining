import csv
import numpy
import sys
import random

# datasets
datasetFile = "../dataset/dataset.csv"
testFile = "../dataset/test.csv"

class Genetic:
	pop = []
	best = []
	dim = 3
	N = 200
	tx_selec = 0.3
	tx_mut = 0.1
	generations = 100
#	
	def fit(self,X,y): 		
		#generating initial population
		self.pop = self.generateInitialPopulation() 

		k = 1
		print("Start evolution")
		while(k <= self.generations):
			results = []
			print("generation", k)
			for i in range(self.N):
				res = self.predict(X,y,self.pop[i]);
				results.append(res)
        	
			# select bests individuals (tx_selec * N)
			bestIndvs = self.selectBestIndvs(self.pop,results)
			
			# best individual of this generation
			best = bestIndvs[0]
			#print("Best individual", best)
			
			# reproduce bests
			self.pop = self.reproduce(bestIndvs)
						
			k += 1
		print("Finished evolution!")
		writeBestInFile(best)
		print("Best Individual Ever", best)
#
	def generateInitialPopulation(self):
		print("Generating inicial population...")
		population = []
		for i in range(self.N):
			indv = numpy.random.uniform(-10,10,self.dim+1)
			population.append(indv.tolist())
		return population
#
	def predict(self,X,y,W):
		net = 0
		soma = 0
		for i in range(len(X)):
			net = numpy.dot(W[1:],X[i]) + W[0]
			Y_est = (1/(1+numpy.e**(-net)))
			soma += self.error(y[i],Y_est)
		return soma
#
	def error(self,Y,Y_est):
		return (Y - Y_est)**2
#       
	def selectBestIndvs(self,pop,results):
		results, pop = zip(*sorted(zip(results,pop)))
		best = []
		N_bests = int(self.tx_selec*self.N)
		for i in range(N_bests):
			best.append(pop[i])
		return best
#
	def reproduce(self, bestIndvs):
		new_pop = bestIndvs
		while(len(new_pop) < self.N):
			# reproduce two random individuals
			indv1 = bestIndvs[numpy.random.randint(0,len(bestIndvs))]
			indv2 = bestIndvs[numpy.random.randint(0,len(bestIndvs))]
			new_pop = self.crossover(new_pop,indv1,indv2)
		return new_pop
#
	def crossover(self,new_pop,indv1,indv2):
		new_indv = []
		for i in range(len(indv1)):
			j = numpy.random.random()
			if (j < 0.5):
				new_indv.append(indv1[i]) 
			else:
				new_indv.append(indv2[i])

		# each individual has a mutate chance
		mut = numpy.random.rand(0,1)
		if (mut < self.tx_mut):
			print("Mutation occurred!")
			new_indv = self.mutate(new_indv)
						
		new_pop.append(new_indv)
						
		return new_pop
#
	def mutate(self,new_indv):
		mut_gene = numpy.random.uniform(-10,10)
		index = numpy.random.randint(0,len(new_indv))
		new_indv[index] = mut_gene
		return new_indv
#

def writeBestInFile(best):
	file = open("best_indv.txt","w")
	file.write(str(best))
	file.close()

#MAIN

print("Reading dataset train file %r..." % datasetFile)
with open(datasetFile) as csvfile:
	reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
	train = list(reader)

X = numpy.array(train).astype('int')
label = X[:,3]

X = numpy.delete(X, 3, 1)
p = Genetic()

p.fit(X,label)

