# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:30:59 2017
solving the TOV equation with different methods and comparing them

implemented rk4, euler, newton over the whole interval

@author: Fabian
"""
###########################Imports########################

import numpy as np
import matplotlib.pyplot as plt
import timeit
import os
import scipy.sparse
import scipy.sparse.linalg


##########################Funktionen#################################
#Equation of state: 
#returns the density for a given pressure value
def rho(p):
	r = 0;
	if(p < 0.):
	    r = 0.
	else:	
	    r = (p/k)**(1./gam) 
	return r

#one term of the first diff eq
def rho_term(p):			
	return (rho(p) + p/c**2)

#one term of the first diff eq
def m_term(r, p, m):
	return (m + 4.*np.pi*r**3*p/c**2)

#one term of the first diff eq
def g_term(r, m):
	return (1.- 2.*G*m / (r * c**2))

#first diff eq
def changeP(r, p, m):	
	if r == 0.:
		return 0.
	else:
		return -G/r**2 * rho_term(p) * m_term(r, p, m) / g_term(r, m)

#second diff eq
def changeM(r, p):
	if r == 0.:
		return 0.
	else:
		return 4.*np.pi*r**2*rho(p)

#return the schwarzschild radius for a mass value
def r_s(m):
	return 2.*G*m/c**2

#pressure derivative of the mass differential equation
def deriv_mass(r, p):
	return 4.*np.pi*r**2*deriv_rho(p)

#pressure derivative of the density
def deriv_rho(p):
	if(p <= 0.):
		return 0.
	else:
		return (p/k)**(1./gam-1.)/(gam*k)


def inv(A):
	det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
	#print("det ", det)
	#print(A[0,0])
	#print(A[1,1])
	#print(A[1,0])
	#print(A[0,1])
	
	B = np.zeros((2,2))
	
	B[0,0] = A[1,1]/det	
	B[0,1] = -A[0,1]/det
	B[1,0] = -A[1,0]/det
	B[1,1] = A[0,0]/det	
	
	#print(B)
	return B
	
	
def residual(l, dr, r, p, m, m_tot):

	R = np.zeros(l-1)	

	R[0] = m[-1]-m_tot	#bc mass is already calculated
	for i in range(1,l-1):
		R[i] = (p[i+1]-p[i-1])/(2*dr) - changeP(r[i], p[i], m[i])
		
	return R
	
	
def residual2(l, dr, r, p, m, m_tot):
	
	R = np.zeros(l-1)	

	
	for i in range(0,l-2):
		R[i] = (p[i+1]-p[i])/dr - 0.5*(changeP(r[i], p[i], m[i]) + changeP(r[i+1], p[i+1], m[i+1]))


	R[-1] = m[-1]-m_tot	#bc mass is already calculated
	return R


#2.5 better array use
def residual25(l, dr, r, p, m, m_tot):
	
	R = np.zeros(l-1)	

	R[:-1] = (p[1:-1]-p[:-2])/dr - 0.5*(changeP(r[:-2], p[:-2], m[:-2]) + changeP(r[1:-1], p[1:-1], m[1:-1]))

	R[-1] = m[-1]-m_tot	#bc mass is already calculated
	return R



def residual3(l, dr, r, p, m, m_tot):

	R = np.zeros(l-1)	

	R[0] = m[-1]-m_tot	#bc mass is already calculated
	for i in range(1,l-1):
		R[i] = (p[i+1]-p[i-1])/(2*dr) - 0.5*(changeP(r[i], p[i], m[i]) + changeP(r[i+1], p[i+1], m[i+1]))
		
	return R
	
	

def jacobian(l, dr, r, p, m):

	#jMat = np.zeros((l-1,l-1))
	jMat = scipy.sparse.lil_matrix((l-1,l-1))	
	
	
	
	for i in range(1,l-1):
		
		jMat[i, i-1] = -1./(2.*dr)
		
		jMat[i, i] = G/(r[i]**2*g_term(r[i],m[i])) * ((deriv_rho(p[i]) + 1./c**2) * m_term(r[i],p[i],m[i]) + rho_term(p[i])*4.*np.pi*r[i]**3/c**2)
		
		
		if i<(l-2):
			jMat[0,i] = 0.5*deriv_rho(p[i]) * (r[i+1]**3-r[i-1]**3) 
			
			jMat[i, i+1] = 1./(2.*dr) 
			
			
			
	jMat[0,l-2] = 0.5*deriv_rho(p[l-2]) * (r[l-1]**3-r[l-3]**3)
	
	jMat[0,0] = 0.5*deriv_rho(p[0]) * (r[1]**3-r[0]**3)
	
	jMat[0] = jMat[0] *4./3.*np.pi
	
	j = jMat.tocsc()
	return jMat
	
	
def jacobian2(l, dr, r, p, m):

	#jMat = np.zeros((l-1,l-1))	
	jMat = scipy.sparse.lil_matrix((l-1,l-1))
	
	for i in range(0,l-2):
		#bc the derivative vanishes for r = 0
		if i == 0:
			jMat[i,i] = -1./dr
		else:
			jMat[i, i] = -1./dr + 0.5*G/(r[i]**2*g_term(r[i],m[i])) * ((deriv_rho(p[i]) + 1./c**2) * m_term(r[i],p[i],m[i]) + rho_term(p[i])*4.*np.pi*r[i]**3/c**2)
		
		jMat[i, i+1] = 1./dr + 0.5*G/(r[i+1]**2*g_term(r[i+1],m[i+1])) * ((deriv_rho(p[i+1]) + 1./c**2) * m_term(r[i+1],p[i+1],m[i+1]) + rho_term(p[i+1])*4.*np.pi*r[i+1]**3/c**2)
		
		
		if i>0:	
			jMat[-1,i] = 0.5*deriv_rho(p[i]) * (r[i+1]**3-r[i-1]**3)
			
	jMat[-1,l-2] = 0.5*deriv_rho(p[l-2]) * (r[l-1]**3-r[l-3]**3)

	jMat[-1,0] = 0.5*deriv_rho(p[0]) * (r[1]**3-r[0]**3)

	jMat[-1] = jMat[-1] *4./3.*np.pi


	j = jMat.tocsc()
	return j
	
	
def jacobian3(l, dr, r, p, m):

	jMat = scipy.sparse.lil_matrix((l-1,l-1))	
	
	for i in range(1,l-1):
		
		jMat[i, i-1] = -1./(2.*dr)
		
		jMat[i, i] = 0.5*G/(r[i]**2*g_term(r[i],m[i])) * ((deriv_rho(p[i]) + 1./c**2) * m_term(r[i],p[i],m[i]) + rho_term(p[i])*4.*np.pi*r[i]**3/c**2)
		
		if i<(l-2):	
		
			jMat[i, i+1] = 1./(2.*dr) + 0.5*G/(r[i+1]**2*g_term(r[i+1],m[i+1])) * ((deriv_rho(p[i+1]) + 1./c**2) * m_term(r[i+1],p[i+1],m[i+1]) + rho_term(p[i+1])*4.*np.pi*r[i+1]**3/c**2)
			
			jMat[0,i] = 0.5*deriv_rho(p[i]) * (r[i+1]**3-r[i-1]**3) 

			
	
	jMat[0,l-2] = 0.5*deriv_rho(p[l-2]) * (r[l-1]**3-r[l-3]**3)
	
	jMat[0,0] = 0.5*deriv_rho(p[0]) * (r[1]**3-r[0]**3)
	
	jMat[0] = jMat[0] *4./3.*np.pi
	
	j = jMat.tocsc()
	return j
	
########################################Start######################################




def newton(m_total, n_zones = 10000, rho_c = 2e14):	



	r_max =  1500000.
	dr = r_max/n_zones
	

	p_c = k*pow(rho_c, gam)
	m_c = 0.

	r = np.zeros(n_zones)
	p = np.zeros(n_zones)
	m = np.zeros(n_zones)
	#dens = np.zeros(n_zones)
	#alpha = np.zeros(n_zones)
	
	p[0] = p_c

	for i in range(1, n_zones):
		r[i] = i*dr
		p[i] = p_c - i*p_c/n_zones
		#m[i] = m[i-1] + 4./3. * np.pi *rho( 0.5*(p[i]+p[i-1]) ) *(r[i]**3 - r[i-1]**3) 
		m[i] = m[i-1] + 4./3. * np.pi *0.5 *(rho(p[i])+rho(p[i-1])) * (r[i]**3-r[i-1]**3)

	p[-1] = 0 	#fixed boundary
	#dens[0] = rho_c
	#alpha[0] = 0.
	
	for t in range(100):
	
		#res = residual3(n_zones, dr, r, p, m, m_total)
		res = residual2(n_zones, dr, r, p, m, m_total)
		#res = residual(n_zones, dr, r, p, m, m_total)

		#j = jacobian3(n_zones, dr, r, p, m)
		j = jacobian2(n_zones, dr, r, p, m)
		#j = jacobian(n_zones, dr, r, p, m)

		#x = np.linalg.solve(j, -res)	
		x = scipy.sparse.linalg.spsolve(j, -res)

		flag = True
		save = r[-1]
		for i in range(n_zones-1):
#chosing whether to cut negative pressure values for iteration purposes

			#p[i] = p[i] + x[i]
			p[i] = max(p[i] + x[i], 0.)
			
			if flag and p[i] == 0:
				flag = False
				save = r[i]			

			if i>0:
				#m[i] = m[i-1] + 4./3. * np.pi *rho( 0.5*(p[i]+p[i-1]) ) *(r[i]**3 - r[i-1]**3)
				m[i] = m[i-1] + 4./3. * np.pi *0.5 *(rho(p[i])+rho(p[i-1])) * (r[i]**3-r[i-1]**3)

			#dens[j+1] = rho(p[j+1])
			#alpha[j+1] = r_s(m[j+1])/r[j+1]
		
#maybe change from rho at middle pressure to middle rho
		#m[-1] = m[-2] + 4./3. * np.pi *rho( 0.5*(p[-1]+p[-2]) ) *(r[-1]**3 - r[-2]**3)
		m[-1] = m[-2] + 4./3. * np.pi *0.5 *(rho(p[-1])+rho(p[-2])) * 				(r[-1]**3-r[-2]**3)		

		print('differenz: ', m_total-m[-1])
		print(m[-1]/solar_mass)
		print(rho(p[0]))

		if m[-1] == m_total:
			print('equal')
			print(t)
			break

		
	
	#plt.plot(r,m)
	#plt.show()

	#plt.plot(r,p)
	#plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
	#plt.xscale('log')
	#plt.show()

	#curr_path = os.path.dirname(os.path.abspath(__file__))
	#path = curr_path + '/'
	#if not os.path.exists(path):
	#	os.makedirs(path)

	#np.savetxt(path + 'Newton_data.txt', np.c_[r, p, m, dens, alpha], delimiter=', ')

	#return (r, p, m, dens, alpha)
	return(save)



#fixed constants in cgs-system
G = 6.6742e-08			#gravitational constant
c = 2.99792458e10		#speed of light
solar_mass = 1.9855e33      	#mass of the sun

#constants for EOS, values can be altered
k = 1.98183e-06             	#polytropic constant
gam = 2.75                 	#adiabatic exponent

n_zones = 10000
m_total = 1.44*solar_mass
rho_c = 2e14


newton(m_total, n_zones, rho_c)


#laenge = 20

#rad = np.zeros(laenge)
#masses = np.zeros(laenge)

#for i in range(laenge):
#	rad[i] = newton(m_total, n_zones, rho_c)
#	masses[i] = m_total
#	m_total = m_total - 0.05*solar_mass


#plt.plot(rad, masses/solar_mass)
#plt.show()



















