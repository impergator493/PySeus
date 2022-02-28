import numpy as np
import matplotlib.pyplot as plt

# def SE(x,y,intc,beta):
#     return (1./len(x))*(0.5)*sum(y - beta * x - intc)**2

# def L1(intc,beta,lam):
#     return lam*(np.abs(intc)+np.abs(beta))

# def L2(intc,beta,lam):
#     return lam*(intc**2 + beta**2)

# N = 100
# x = np.random.randn(N)
# y = 2 * x + np.random.randn(N)

# beta_N = 100
# beta = np.linspace(-40,40,beta_N)
# intc = 0.0

# SE_array = np.array([SE(x,y,intc,i) for i in beta])
# L1_array = np.array([L1(intc,i,lam=30) for i in beta])
# L2_array = np.array([L2(intc,i,lam=1) for i in beta])

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1,1,1)
# #ax1.plot(beta,SE_array,label='Squared Error')
# ax1.plot(beta,L1_array,label='L1 norm')
# ax1.plot(beta,L2_array,label='Squared L2 norm')
# plt.title('Comparison of loss function with L1 and squared L2 norm')
# plt.legend()
# fig1.show()

# objective_L1 = SE_array + L1_array
# objective_L2 = SE_array + L2_array


# fig2 = plt.figure()
# ax2 = fig2.add_subplot(1,1,1)
# ax2.plot(beta,objective_L1,label='L1')
# ax2.plot(beta,objective_L2,label='L2')
# plt.title('Objective Function (Squared Error + L1 or L2 norm)')
# plt.legend()
# fig2.show()


######
beta_N = 100
beta1 = np.linspace(-2.5,2.5,beta_N)
beta2 = np.linspace(-2.4,1.7,beta_N)
intc = 0.0

Nonconvex = np.array(0.65*beta2**4+beta2**3-2*beta2**2-2*beta2+3)
Convex = np.array(0.5*beta1**2+1)

fig1 = plt.figure()
st = fig1.suptitle("Minimization of non-convex and convex cost function")
ax1 = fig1.add_subplot(1,2,1)
ax1.plot(beta1,Nonconvex,label='Non-convex function')
plt.plot(-1.77, 1.12, "o")
plt.annotate('Local min '+ r'$\neq$'+' global min', (-2.8, 0.5))
plt.ylabel("Cost")
plt.xlim((-3,3))
plt.ylim((0,5))
plt.legend()
ax2 = fig1.add_subplot(1,2,2)
ax2.plot(beta2,Convex,label='Convex function')
plt.plot(-0.37, 1.00, "o")
plt.annotate("Local min = global min", (-2, 0.5))
plt.xlim((-3,3))
plt.ylim((0,5))
plt.legend()
fig1.show()

fig2 = plt.figure()
