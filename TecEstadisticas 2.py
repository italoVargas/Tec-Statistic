#            ITALO VARGAS QUIROZ

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import *
import scipy as sp




#Este codigo lee los datos agrupados en  columnas (expeto primera fila de titulo ) de un archivo exel llamado Datos.xls que debe estar en la misma caarpeta que el archivo , considerar instalar las libreriaas aantes mostradas para que se puedaa ejecutar. 
def GE(Datos):

	doc = pd.read_excel(Datos)
	datadic = doc.to_dict()
	darray = np.array(doc)#,dtype=float)
#	---  elimina los NAN por 0 --- 
	#dataarray = np.where(np.isnan(darray),0,darray)
	#dataarray = darray[np.logical_not(np.isnan(darray))]
	Dataarray=np.transpose(darray)
	Filas=len(darray)
	Columnas=len(Dataarray)
	#print(datadic)

	#print(Dataarray)
	Titulo='Análisis Estádistico de Datos'
	N=[]
	Promedio=[]
	Media=[]
	Moda=[]

	Sigma=[]
	Varianza=[]
	Descripcion=[]
	
	#Datarango=np.array((Columnas,Filas))#np.array((Columnas,Filas))
	Datarango=[]
	DRango=[]
	Dnorm=[]
	DistNorm=[]
	LIC=[]
	LSC=[]
	LIE=2.5	
	LSE=5.5
	NBarrasHist=6

	#Matriz=np.empty((Columnas,Filas),dtype=float)
	#print(Matriz)

#	---  Quita los nan  --- 
	for columna in range(Columnas):
		for fila in range(Filas):
			dato=Dataarray[columna][fila]
			#Matriz[columna][fila].append(dato)
			#Matriz [columna][fila] = darray[np.isfinite(darray)]

			#print(dato)
			if dato == 'nan':
				print('siahay')
			#	np.delete(Dataarray,[columna,fila])
				

			
			#sigma= np.sqrt(np.nansum(((Dataarray[columna][fila])-(Promedio[columna])**2)/(np.len(Dataarray[columna]))))
			#Sigma=np.append(Sigma,sigma)

#---------- Guarda poor columnas-------
	#for i in range(len(darray)

	for columna in range(Columnas):
	#--- Por columna saca el promedio--
		promedio=np.nanmean(Dataarray[columna])
		Promedio=np.append(Promedio,promedio)

		media=np.nanmedian(Dataarray[columna])
		Media=np.append(Media,media)

		moda=stats.mode(Dataarray[columna])
		Moda=np.append(Moda,moda)

		sigma=pd.Series((Dataarray[columna])).std()
		Sigma=np.append(Sigma,sigma)

		varianza=pd.Series((Dataarray[columna])).var()
		Varianza=np.append(Varianza,varianza)


		#descripcion=sp.stats.describe(Dataarray[columna])		
		#Descripción=np.append(Descripcion,descripcion

#PARA SABER CUANTOS DATOS SIN NAN hay en una lista
		n=Dataarray[columna].size - np.count_nonzero(np.isnan(Dataarray[columna]))
		N=np.append(N,n)

#------------------ Graficar HISTOGRAMA -------- 

		plt.figure('Histogrma',figsize=(10, 9))
#######que se vea grande las graficas siguientes de aqui 
		#plt.style.use('fivethirtyeight')
		#plt.style.use('classic')
		#plt.style.use('bmh')
		plt.style.use('seaborn-ticks')
		
		#colors=plt.cm.BuPu(np.linspace(0,0.5,len(Dataarray[columna])))
		#plt.hist(Dataarray[columna],color=colors[Filas-1], ec='black')
		
		Histograma=plt.hist(Dataarray[columna],NBarrasHist
							,color="skyblue", ec='black',facecolor='g'
							,alpha=0.35)#,histtype='bar',density=True)#,rwidth=.5,normed=True,stacked=False)
		Histograma
#Histograma en 2D		
		#plt.hist2d(Dataarray[columna],Dataarray[columna+1],color="skyblue", ec='black')#deben de ser Xy Y del mismo tamaño


		#DNorm=np.random.normal(Promedio[columna],Sigma[columna],Dnorm)
		#x=np.arange(0,Dnorm,1)
		#plt.plot(x,DNorm,color="skyblue")
		#x=[]
		
#distrubucion normal
		
		
		'''
		for fila in range(Filas):
		# Esto es lo mismo que lo de abajo sp.stats.norm.pdf
			y=1/(Sigma[columna]*np.sqrt(2*np.pi))*np.exp((-1)*(((Dataarray[columna][fila]-Promedio[columna])**2)/(2*Sigma[columna]**2)))
			DistNorm=np.append(DistNorm,y)
			DistNorm
		x=np.arange(0,len(DistNorm),1)
		plt.hist(DistNorm,alpha=1)
		DistNorm=[]
		

			#print(y,DistNorm)

		distnorm = sp.stats.norm.pdf(Dataarray[columna],Promedio[columna],Sigma[columna])
		np.append(DistNorm,distnorm)
		print(DistNorm)
		x=np.arange(0,len(DistNorm),1)
		plt.plot(x,DistNorm)
		DistNorm=[]
		
		
		#best_fit_line = sp.stats.norm.pdf(Histograma, Promedio[columna], Sigma[columna])

		#plt.plot(Histograma,best_fit_line)

		'''
# ###########PROMEDIO
		plt.axvline(x=Promedio[columna],linewidth=2, color='m',label=u'\u00B5'+'= '+(str(round(Promedio[columna],2))))

		#textos promedios
	#	plt.text((round(Promedio[columna],2)), 0, r'$\mu=$'+str(round(Promedio[columna],3)), r'$\sigma=$'+str(round(Sigma[columna],3)))
		#plt.figtext(.5,.9 , round(Promedio[columna],3))
		#plt.figtext(float(Promedio[columna]),0 , str(Promedio[columna]))
		#plt.subplots.annotate(Promedio[columna], xy=(Promedio[columna],0)),



# #########Sigmas Histogramas
		plt.axvline(Promedio[columna]+Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = u'\u03C3' +'= ' +(str(round(Sigma[columna],2)))+ '  ;  1'u'\u03C3' +'= ' +(str(round(Promedio[columna]+Sigma[columna],2))))
		plt.axvline(Promedio[columna]+2*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '2'+u'\u03C3' +'= ' +(str(round(Promedio[columna]+2*Sigma[columna],2))))

		plt.axvline(Promedio[columna]+3*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '3'+u'\u03C3' +'= ' +(str(round(Promedio[columna]+3*Sigma[columna],2))))


		#Negativas
		plt.axvline(Promedio[columna]-Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '-1'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-Sigma[columna],2))))

		plt.axvline(Promedio[columna]-2*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '-2'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-2*Sigma[columna],2))))

		plt.axvline(Promedio[columna]-3*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '-3'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-3*Sigma[columna],2))))




# ###############LIMITES ESPECIFICADOS -
		plt.axvline(x=LIE,linewidth=2.5, color='r',linestyle='-.',label='LIE= '+(str(LIE)))
		plt.axvline(x=LSE,linewidth=2.5, color='r',linestyle='-.',label='LSE= '+(str(LSE)))

##############Limites de Control
		LSC= Promedio[columna]+4*Sigma[columna] 
		LIC= Promedio[columna]-4*Sigma[columna] 
		plt.axvline(x=LSC,linewidth=2, linestyle=':', color='g', label = "LSC= "+(str(round(LSC,2))))#, marker='x')
		plt.axvline(x=LIC,linewidth=2, linestyle=':', color='g', label = "LIC= "+(str(round(LIC,2))))#, marker='x')
		#plt.legend(['0-10','10-100','100-500','500+','10-100','100-500','500+','10-100','100-500','500+','10-100','100-500','500+','10-100','100-500','500+'],loc='best')
		
		plt.legend(loc='best')
		plt.legend()
		plt.grid(True)
		plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.15)
		plt.minorticks_on()
		#plt.xlim(LIE-1, LSE+1)
		#plt.ylim(0, 0.03)
		plt.xlabel('COLUMNA'+str(columna+1))
		plt.ylabel('Frecuencia')	
		plt.title('Histograma')
		plt.savefig('Histograma'+str(columna+1))
		plt.show()



#---------- Graficar Control  -------- 
		plt.figure('Grafico de Control',figsize=(10, 9))
		plt.style.use('seaborn-ticks')
		x=np.arange(0,len(Dataarray[columna]),1)
		plt.scatter(x,Dataarray[columna],color="skyblue", ec='blue')#,cmap='YlGnBu')
		plt.plot(x,Dataarray[columna],color="skyblue")

#            PROMEDIO
		plt.axhline(y=Promedio[columna],linewidth=2, color='m',label=u'\u00B5'+'= '+(str(round(Promedio[columna],2))))


#              Sigmas GCOntrol
		plt.axhline(Promedio[columna]+Sigma[columna],linewidth=1.5, linestyle='--', color='c', label =  u'\u03C3' +'= ' +(str(round(Sigma[columna],2)))+ '  ;  1'u'\u03C3' +'= ' +(str(round(Promedio[columna]+Sigma[columna],2))))
		plt.axhline(Promedio[columna]+2*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '2'+u'\u03C3' +'= ' +(str(round(Promedio[columna]+2*Sigma[columna],2))))
		plt.axhline(Promedio[columna]+3*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '3'+u'\u03C3' +'= ' +(str(round(Promedio[columna]+3*Sigma[columna],2))))
		#Negativas
		plt.axhline(Promedio[columna]-Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '-1'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-Sigma[columna],2))))
		plt.axhline(Promedio[columna]-2*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '-2'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-2*Sigma[columna],2))))

		plt.axhline(Promedio[columna]-3*Sigma[columna],linewidth=1.5, linestyle='--', color='c', label = '-3'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-3*Sigma[columna],2))))




#           LIMITES ESPECIFICADOS

		plt.axhline(y=LSE,linewidth=2, linestyle='-', color='r',label='LIE= '+(str(LIE)))
#, marker='x')
		plt.axhline(y=LIE,linewidth=2, linestyle='-', color='r',label='LSE= '+(str(LSE)))

#             Limites de Control
		LSC= Promedio[columna]+4*Sigma[columna] 
		LIC= Promedio[columna]-4*Sigma[columna] 
		plt.axhline(y=LSC,linewidth=2, linestyle=':', color='g', label = "LSC= "+(str(round(LSC,2))))#, marker='x')
		plt.axhline(y=LIC,linewidth=2, linestyle=':', color='g', label = "LIC= "+(str(round(LIC,2))))#, marker='x')


		plt.legend(loc='best')
		plt.legend()
		plt.grid()
		plt.minorticks_on()		
		plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.15)
		plt.xlabel('COLUMNA'+str(columna+1))
		plt.ylabel('Valor')	
		plt.title('Gráfico de Control')
		plt.savefig('GControl'+str(columna+1))
		plt.show()
		

#------ Graficar Series de tiempo  -------- 
		plt.figure('Gráfico de Series de Tiempo',figsize=(10, 9)).add_subplot(2,1,1)
		plt.style.use('seaborn-ticks')
		x=np.arange(0,len(Dataarray[columna]),1)
		plt.scatter(x,Dataarray[columna],color="skyblue", ec='blue')
		plt.plot(x,Dataarray[columna],color="skyblue")
		plt.grid()
		plt.minorticks_on()
		plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.15)
		plt.xlabel('Tiempo')
		plt.ylabel('COLUMNA'+str(columna+1))
		plt.title('Gráfico de Series de Tiempo')

		
#            PROMEDIO
		plt.axhline(y=Promedio[columna],linewidth=2, color='m',label=u'\u00B5'+'= '+(str(round(Promedio[columna],2))))


#              Sigmas GCOntrol
		plt.axhline(Promedio[columna]+Sigma[columna],linewidth=1, linestyle='--', color='c', label =  u'\u03C3' +'= ' +(str(round(Sigma[columna],2)))+ '  ;  1'u'\u03C3' +'= ' +(str(round(Promedio[columna]+Sigma[columna],2))))
		plt.axhline(Promedio[columna]+2*Sigma[columna],linewidth=1, linestyle='--', color='c', label = '2'+u'\u03C3' +'= ' +(str(round(Promedio[columna]+2*Sigma[columna],2))))

		plt.axhline(Promedio[columna]+3*Sigma[columna],linewidth=1, linestyle='--', color='c', label = '3'+u'\u03C3' +'= ' +(str(round(Promedio[columna]+3*Sigma[columna],2))))


		#Negativas
		plt.axhline(Promedio[columna]-Sigma[columna],linewidth=1, linestyle='--', color='c', label = '-1'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-Sigma[columna],2))))

		plt.axhline(Promedio[columna]-2*Sigma[columna],linewidth=1, linestyle='--', color='c', label = '-2'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-2*Sigma[columna],2))))

		plt.axhline(Promedio[columna]-3*Sigma[columna],linewidth=1, linestyle='--', color='c', label = '-3'+u'\u03C3' +'= ' +(str(round(Promedio[columna]-3*Sigma[columna],2))))




#           LIMITES ESPECIFICADOS

		plt.axhline(y=LSE,linewidth=2, linestyle='-', color='r',label='LIE= '+(str(LIE)))
		plt.axhline(y=LIE,linewidth=2, linestyle='-', color='r',label='LSE= '+(str(LSE)))

#             Limites de Control
		LSC= Promedio[columna]+4*Sigma[columna] 
		LIC= Promedio[columna]-4*Sigma[columna] 
		plt.axhline(y=LSC,linewidth=2, linestyle=':', color='g', label = "LSC= "+(str(round(LSC,2))))#, marker='x')
		plt.axhline(y=LIC,linewidth=2, linestyle=':', color='g', label = "LIC= "+(str(round(LIC,2))))#, marker='x')#, marker='x')



		plt.legend(loc='best', fontsize='x-small')


		#plt.savefig('GSerTiempo'+str(columna+1))
		#plt.show()

#------ Grafica Rango -------- 
		plt.figure('Gráfico de Series de Tiempo').add_subplot(2,1,2)
		#plt.figure('Gráfico de Rango')
		plt.style.use('seaborn-ticks')
		for fila in range(Filas-1):
			dsiguiente=Dataarray[columna][fila+1]
			rango=Dataarray[columna][fila]-dsiguiente
			#print(rango)
			Datarango=np.append(Datarango,rango)
			#print(rango,Datarango,type(Datarango))

		#PARA SABER CUANTOS DATOS SIN NAN hay en una lista

		#DRango=Datarango.size - np.count_nonzero(np.isnan(Datarango))

		x=np.arange(0,len(Datarango),1)
		#print(x,Datarango)
		plt.scatter(x,abs(Datarango),color="skyblue", ec='blue')
		plt.plot(x,abs(Datarango),color="skyblue")
		plt.axhline(np.nanmin(abs(Datarango)),linewidth=2, linestyle=':', color='g', label = 'MIN= '+(str(round(np.nanmin(abs(Datarango)),1))))#, marker='x')
		plt.axhline(np.nanmax(abs(Datarango)),linewidth=2, linestyle=':', color='g', label = 'MAX= '+(str(round(np.nanmax(abs(Datarango)),1))))#, marker='x')
		plt.legend(loc='best')#, fontsize='small')
		plt.minorticks_on()

		plt.grid()
		plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.15)
		plt.xlabel('Tiempo')
		plt.ylabel('Rango')
		plt.title('Gráfico de Rango Móvil')
		plt.savefig('SeriesDeTiempo-RangoM'+str(columna+1))
		plt.show()
		Datarango=[]



		

		


	Estadistica=(Promedio,Media,Moda,Sigma,Varianza,N)



	print('Promedio por columna= '+str(Estadistica[0]))
	print('Media por columna= '+str(Estadistica[1]))
	print('Moda por columna,datdeter= '+str(Estadistica[2]))
	print('Dev.Estandar por columna= '+str(Estadistica[3]))
	print('Varianza por columna= '+str(Estadistica[4]))
	print('Numero de datoos por columna= '+str(Estadistica[5]))
	#print(Descripcion)

	
print(GE('Datos.xls'))







