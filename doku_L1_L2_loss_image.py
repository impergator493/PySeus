import numpy as np
import matplotlib.pyplot as plt

def SE(x,y,intc,beta):
    return (1./len(x))*(0.5)*sum(y - beta * x - intc)**2

def L1(intc,beta,lam):
    return lam*(np.abs(intc)+np.abs(beta))

def L2(intc,beta,lam):
    return lam*(intc**2 + beta**2)

N = 100
x = np.random.randn(N)
y = 2 * x + np.random.randn(N)

beta_N = 100
beta = np.linspace(-40,40,beta_N)
intc = 0.0

SE_array = np.array([SE(x,y,intc,i) for i in beta])
L1_array = np.array([L1(intc,i,lam=30) for i in beta])
L2_array = np.array([L2(intc,i,lam=1) for i in beta])

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
#ax1.plot(beta,SE_array,label='Squared Error')
ax1.plot(beta,L1_array,label='L1 norm')
ax1.plot(beta,L2_array,label='Squared L2 norm')
plt.title('Comparison of loss function with L1 and squared L2 norm')
plt.legend()
fig1.show()

objective_L1 = SE_array + L1_array
objective_L2 = SE_array + L2_array


fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(beta,objective_L1,label='L1')
ax2.plot(beta,objective_L2,label='L2')
plt.title('Objective Function (Squared Error + L1 or L2 norm)')
plt.legend()
fig2.show()



