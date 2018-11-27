import numpy as np
import numpy.random as npr
import scipy.special as sps


class vi_lda(object):

	def __init__(self, X, K=10):
		# X = matrix where rows:documents and columns:words
		self.X = X

		# K = number of topics
		self.K = K
		self.initialize_hyperparams()


	def initialize_hyperparams(self):
	    N, M = self.X.shape

	    # initialize variational parameters
	    self.alpha = npr.gamma(self.K, 1/self.K, (self.K)).astype(np.float32)
	    self.eta = npr.gamma(self.K, 1/self.K, (M)).astype(np.float32)
	    self.current_lambda = npr.gamma(self.K, 1/self.K, (self.K, M)).astype(np.float32)
	    self.current_gamma = npr.gamma(self.K, 1/self.K, (N, self.K)).astype(np.float32)
	    self.current_phi = np.full((N, M, self.K), 1/self.K).astype(np.float32)

	"""
	Update the variational parameters phi, gamma, and lambda individually,
	while holding the other variational parameters fixed. 
	""" 
	def update_phi(self):
	    self.Elog_beta = (sps.digamma(self.current_lambda) - 
			      sps.digamma(self.current_lambda.sum(axis=1)[:, None])).astype(np.float32)
	    self.Elog_theta = (sps.digamma(self.current_gamma) - 
			       sps.digamma(self.current_gamma.sum(axis=1)[:, None])).astype(np.float32)

	    self.Elog_beta_ = (self.Elog_beta.T[None, :, :] * self.X[:, :, None]).astype(np.float32)
	    self.Elog_theta_ = (self.Elog_theta[:, None, :] * self.X[:, :, None]).astype(np.float32)

	    self.phi = np.exp(self.Elog_beta_ + self.Elog_theta_).astype(np.float32)
	    self.current_phi = (self.phi/self.phi.sum(axis=2, keepdims=True)).astype(np.float32)


	def update_gamma(self):
		self.current_gamma = np.add(self.alpha[None, :], self.current_phi.sum(axis=1))


	def update_lambda(self):
		for k in range(self.K):
			update = np.multiply(self.X, self.current_phi[:, :, k]).sum(axis=0)
			self.current_lambda[k] = self.eta + update


	# calculate the evidence lower bound (ELBO)
	def expected_log_joint(self):
	    beta = np.outer((self.eta-1), self.Elog_beta.sum(axis=1)).sum(1).sum()
	    theta = np.outer((self.alpha-1), self.Elog_theta.sum(axis=1)).sum(1).sum()
	    phi_theta = np.einsum('ijk,ik', self.current_phi, self.Elog_theta).sum()
	    phi_beta = np.einsum('ijk, kj', self.current_phi, self.Elog_beta).sum()

	    self.exp_log_joint = beta + theta + phi_theta + phi_beta


	def entropy(self):
	    log_q_beta = (sps.loggamma(self.current_lambda.sum(axis=1)) - \
			  sps.loggamma(self.current_lambda).sum(axis=1) + \
			  ((self.current_lambda-1)*self.Elog_beta).sum(axis=1)).sum(axis=0)

	    log_q_theta = (sps.loggamma(self.current_gamma.sum(axis=1)) - \
			   sps.loggamma(self.current_gamma).sum(axis=1) + \
			   ((self.current_gamma-1)*self.Elog_theta).sum(axis=1)).sum(axis=0)

	    log_q_z = -np.einsum('ijk, ijk', self.current_phi, np.log(self.current_phi))

	    self.entropy = log_q_beta + log_q_theta + log_q_z


	def compute_elbo(self):
	    self.elbo = self.exp_log_joint - self.entropy


	# run variational inference with set number of iterations
	def run_vi_fixed_iters(self, iters=100, interval=10):
		elbo = -1e12
		elbo_delta = 0
		self.elbos = list()

		for i in range(iters):
			if i%interval == 0:
			    print("at iteration " + str(i) + ", latest delta: " + str(elbo_delta))
			self.update_phi()
			self.update_gamma()
			self.update_lambda()
			self.expected_log_joint()
			self.entropy()
			self.compute_elbo()
			elbo_delta = self.elbo - elbo
			self.elbos.append(self.elbo)
			elbo = self.elbo

	# TODO: create function to run variational inference until ELBO converges
