import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.utils import spectral_norm
from torch.nn import init

# BCQ code from the author's BCQ implementation: https://github.com/sfujim/BCQ/tree/master/continuous_BCQ
# for CGAN, I modified some code from these sources:
	# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/e9c8374ecc202120dc94db26bf08a00f/dcgan_faces_tutorial.ipynb
	# https://github.com/nbertagnolli/pytorch-simple-gan
# for WGAN-GP I used the training procedure from here: https://github.com/znxlwm/pytorch-generative-model-collections/blob/master/WGAN_GP.py
# for RND, I modified code from here: https://github.com/wizdom13/RND-Pytorch/blob/master/model.py


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400) #bcq has 400
		self.l2 = nn.Linear(400, 300) #bcq has 400, 300
		self.l3 = nn.Linear(300, action_dim) #bcq has 300, action_dim
		self.max_action = max_action
		self.phi = phi

	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a)) #phi is hyperparameter to add better noise to action
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400) #400
		self.l2 = nn.Linear(400, 300) #400, 300
		self.l3 = nn.Linear(300, 1) #300, 1

		self.l4 = nn.Linear(state_dim + action_dim, 400) #400
		self.l5 = nn.Linear(400, 300) #400, 300
		self.l6 = nn.Linear(300, 1) #300, 1

	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Generator: given state and some noise, produce an action
class GeneratorBCQ(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_size=64):
		super(GeneratorBCQ, self).__init__()
		self.action_dim = action_dim
		self.max_action = max_action
		self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, action_dim)

	def forward(self, x, a):
		x = torch.cat([x, a], dim=1)
		x = F.leaky_relu(self.l1(x))
		x = F.leaky_relu(self.l2(x))
		x = torch.tanh(self.l3(x)) * self.max_action

		return x

	def generate_noise(self, num_samples):
		# only pointer I saw for generating noise for GAN is to use gaussian instead of uniform, not sure if other tips
		noise = torch.normal(0, .3, size=(num_samples, self.action_dim)).clamp(-1., 1.)
		return noise


# given state and either a Generator action (0) or true action (1) learn to discriminate
class DiscriminatorBCQ(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size=64, use_wgan=False, use_sn_gan=False, output_size=1):
		super(DiscriminatorBCQ, self).__init__()
		self.use_wgan = use_wgan

		if use_sn_gan:
			# shared layers
			self.l1 = spectral_norm(nn.Linear(state_dim + action_dim, hidden_size))
			self.l2 = spectral_norm(nn.Linear(hidden_size, hidden_size))
		else:
			# shared layers
			self.l1 = nn.Linear(state_dim + action_dim, hidden_size)
			self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, output_size)

	def forward(self, x, a):
		x = torch.cat([x, a], dim=1)
		x = F.leaky_relu(self.l1(x))
		x = F.leaky_relu(self.l2(x))
		if self.use_wgan:
			x = self.l3(x)
		else:
			x = torch.sigmoid(self.l3(x))

		return x


# Generator-Actor combined NN
# Generator: given state and some noise, produce an action
# actor: perturb action with a behavior constraint
class GeneratorActorBCQ(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(GeneratorActorBCQ, self).__init__()
		self.action_dim = action_dim
		self.max_action = max_action
		self.phi = phi
		# shared layers
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		# generator
		self.g1 = nn.Linear(300, action_dim)
		# actor
		self.a1 = nn.Linear(300, action_dim)

	def forward(self, x, a, p=None):
		# shared
		x = torch.cat([x, a], dim=1)
		x = F.leaky_relu(self.l1(x))
		x = F.leaky_relu(self.l2(x))
		# generator
		g = torch.tanh(self.g1(x)) * self.max_action
		# actor
		if p is not None:
			a = p * self.max_action * torch.tanh(self.a1(x)) # perturb action based on predictor model
			a = (a + g).clamp(-self.max_action, self.max_action)
		else:
			a = self.phi * self.max_action * torch.tanh(self.a1(x)) # perturb action based on phi
			a = (a + g).clamp(-self.max_action, self.max_action)

		return g, a

	def generate_noise(self, num_samples):
		# only pointer I saw for generating noise for GAN is to use gaussian instead of uniform, not sure if other tips
		noise = torch.normal(0, .3, size=(num_samples, self.action_dim)).clamp(-1., 1.)
		return noise


# Discriminator Critic combined model
# discriminator: given state and either a Generator action or true action learn to discriminate
# critic: estimated value of the state action combination
# two critic heads to take more conservative estimates
class DiscriminatorCriticBCQ(nn.Module):
	def __init__(self, state_dim, action_dim, use_wgan=False, use_sn_gan=False, d_output_size=1, c_output_size=1):
		super(DiscriminatorCriticBCQ, self).__init__()
		self.use_wgan = use_wgan

		if use_sn_gan:
			# shared layers
			self.l1 = spectral_norm(nn.Linear(state_dim + action_dim, 400))
			self.l2 = spectral_norm(nn.Linear(400, 300))
		else:
			# shared layers
			self.l1 = nn.Linear(state_dim + action_dim, 400)
			self.l2 = nn.Linear(400, 300)
		# discriminator
		self.d1 = nn.Linear(300, d_output_size)
		# critic
		self.c1 = nn.Linear(300, c_output_size)
		self.c2 = nn.Linear(300, c_output_size)

	def forward(self, x, a):
		# shared layers
		x = torch.cat([x, a], dim=1)
		x = F.leaky_relu(self.l1(x))
		x = F.leaky_relu(self.l2(x))
		# discriminator
		if self.use_wgan:
			d = self.d1(x)
		else:
			d = torch.sigmoid(self.d1(x))
		# critic
		c1 = self.c1(x)
		c2 = self.c2(x)

		return d, c1, c2

	def critic_vf(self, x, a):
		x = torch.cat([x, a], dim=1)
		x = F.leaky_relu(self.l1(x))
		x = F.leaky_relu(self.l2(x))
		c1 = self.c1(x)

		return c1


# not really an RND but used sort of the same way:
# fix target network and train predictor network to predict target network's output
# typically RND uses next observations as inputs and differences between network outputs to drive curiosity
# here I use current state and action as inputs and differences between outputs to modify action perturbation (train time) and selection (test time)
class RNDModel(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size=32):
		super(RNDModel, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.target = nn.Sequential(
			nn.Linear(state_dim+action_dim, hidden_size),
			nn.ELU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ELU(),
			nn.Linear(hidden_size, state_dim),
		)

		self.predictor = nn.Sequential(
			nn.Linear(state_dim + action_dim, hidden_size),
			nn.ELU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ELU(),
			nn.Linear(hidden_size, state_dim),
		)

		# Initialize the weights and biases
		for p in self.modules():
			if isinstance(p, nn.Linear):
				init.orthogonal_(p.weight, np.sqrt(2))
				p.bias.data.zero_()

		# Set that target network is not trainable
		for param in self.target.parameters():
			param.requires_grad = False

	def forward(self, x, a): # actual RND uses next observation
		x = torch.cat([x, a], dim=1)
		target_feature = self.target(x)
		predict_feature = self.predictor(x)

		return predict_feature, target_feature


class BCQGAN(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05,
				 ac_lr=1e-3, gan_lr=1e-3, hidden_size=64, use_sngan=False, use_ac=True, use_max_a=True, use_wgan=False, wgan_d_iter=5):
		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda #AWAC paper says this might not be needed
		self.device = device
		self.gp_lambda = 10 # gradient penalty lambda
		self.use_sn_gan = use_sngan
		self.use_wgan = use_wgan
		self.wgan_d_iter = wgan_d_iter
		self.use_ac = use_ac

		# use actor critic architecture with generator and discriminator or G-A and D-C combined models
		if self.use_ac:
			self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
			self.actor_target = copy.deepcopy(self.actor)
			self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ac_lr)
			self.critic = Critic(state_dim, action_dim).to(device)
			self.critic_target = copy.deepcopy(self.critic)
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=ac_lr)

			# Generator and Discriminator for the GAN
			self.generator = GeneratorBCQ(state_dim, action_dim, max_action, hidden_size=hidden_size).to(device)
			self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gan_lr)
			self.discriminator = DiscriminatorBCQ(state_dim, action_dim, hidden_size=hidden_size, use_wgan=self.use_wgan, use_sn_gan=use_sngan).to(device)
			self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=gan_lr)
			self.gan_loss = nn.BCELoss()
		else:
			# combine Actor, Critic, Generator and Discriminator and use shared layers with different output layers
			self.generator = GeneratorActorBCQ(state_dim, action_dim, max_action, phi=phi).to(device)
			self.generator_target = copy.deepcopy(self.generator)
			self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gan_lr)
			self.discriminator = DiscriminatorCriticBCQ(state_dim, action_dim, use_wgan=self.use_wgan, use_sn_gan=use_sngan).to(device)
			self.discriminator_target = copy.deepcopy(self.discriminator)
			self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=gan_lr)
			self.gan_loss = nn.BCELoss()

		# RND related things
		self.use_max_a = use_max_a
		self.rnd_percentile_list = []  # store results of rnd based on percentile
		self.rnd_percentile = 30  # percentile cut off for selecting action, lower percentile means closer agreement between two RND heads
		self.rnd_cutoff_value = None
		if not self.use_max_a:
			self.rnd = RNDModel(state_dim, action_dim, hidden_size=64).to(device)
			self.rnd_loss = nn.MSELoss()
			self.rnd_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=gan_lr)

	def select_action(self, state):
		# select action at evaluation time
			# default uses argmax from best critic (or D-C) predicted action
			# alternative method uses RND to predict an seen action
		with torch.no_grad():
			# state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			# action = self.actor(state, self.vae.decode(state))
			# q1 = self.critic.q1(state, action)
			# ind = q1.argmax(0)
			repeat_value = 100
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(repeat_value, 1).to(self.device)
			noise = self.generator.generate_noise(repeat_value).to(self.device)
			if self.use_ac:
				generator_action = self.generator(state, noise)
				action = self.actor(state, generator_action)
				q1 = self.critic.q1(state, action)
			else:
				_, action = self.generator(state, noise)
				q1 = self.discriminator.critic_vf(state, action)

			if not self.use_max_a:
				# use RND to select best action above some threshold
				if self.rnd_cutoff_value is not None:
					# get output of RND network
					pred_feature, tgt_feature = self.rnd(state, action)
					# get differences between predicted and RND output
					rnd_differences = (pred_feature - tgt_feature).pow(2).sum(dim=1).reshape(-1, 1)
					# concat critic prediction, rnd differences, and actions
					selection_tensor = torch.cat([q1, rnd_differences, action], axis=1)
					cutoff_tensor_value = torch.tensor(self.rnd_cutoff_value).float().to(self.device)
					# only keep actions less than a threshold. self.rnd_cutoff_value is difference between two RND heads, want something smaller thus more seen
					selection_tensor = selection_tensor[selection_tensor[:, 1] < cutoff_tensor_value]
					# check if still some possible actions to select
					if selection_tensor.size()[0] > 0:
						# take argmax of remaining values q values
						ind = selection_tensor[:, 0].argmax()
						# select corresponding action
						action_out = selection_tensor[ind, 2:]
						return action_out.cpu().data.numpy().flatten()

			# choose argmax action
			ind = q1.argmax(0)
			return action[ind].cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations, batch_size=100):
		self.rnd_percentile_list = []

		if self.device != "cpu":
			torch.cuda.empty_cache()

		# debugging
		debug_gen_loss = 0.
		debug_dis_loss = 0.
		debug_dis_t_loss = 0.
		debug_dis_f_loss = 0.
		debug_dis_gp_loss = 0.
		debug_critic_loss = 0.
		debug_actor_loss = 0.
		debug_sampled_actions_list = []
		debug_next_actions_list = []
		debug_d_output_list = []

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			if self.use_ac:
				if self.use_wgan:
					noise = self.generator.generate_noise(batch_size).to(self.device)
					generator_actions = self.generator(state, noise)

					# train discriminator real
					self.discriminator_optimizer.zero_grad()
					true_discriminator_out = self.discriminator(state, action)
					true_discriminator_loss = -torch.mean(true_discriminator_out)
					# train discriminator fake
					gen_for_dis_out = self.discriminator(state, generator_actions)  # not sure if i want to detach here
					gen_dis_loss = torch.mean(gen_for_dis_out)
					# gradient penalty
					alpha = torch.rand((batch_size, self.action_dim)).to(self.device)
					x_hat = alpha * action.data + (1. - alpha) * generator_actions.data
					x_hat.requires_grad = True
					pred_hat = self.discriminator(state, x_hat)
					gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
						 create_graph=True, retain_graph=True, only_inputs=True)[0]
					gradient_penalty = self.gp_lambda * (
								(gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

					discriminator_loss = true_discriminator_loss + gen_dis_loss + gradient_penalty
					discriminator_loss.backward()
					self.discriminator_optimizer.step()
					# debugging
					debug_dis_loss += discriminator_loss.item()
					debug_dis_t_loss += true_discriminator_loss.item()
					debug_dis_f_loss += gen_dis_loss.item()
					debug_dis_gp_loss += gradient_penalty.item()

					if (it + 1) % self.wgan_d_iter == 0:
						# train generator
						self.generator_optimizer.zero_grad()
						generator_actions = self.generator(state, noise)  # only use the one batch of noise created above
						gen_dis_out = self.discriminator(state, generator_actions)
						generator_loss = -torch.mean(gen_dis_out)
						generator_loss.backward()
						self.generator_optimizer.step()
						debug_gen_loss += generator_loss.item()
						debug_d_output_list.append(gen_dis_out[0].cpu().data.numpy())
				else:
					# train D and G, train D first, backwards on each part then do G
					self.discriminator_optimizer.zero_grad()
					# true_labels = torch.ones((batch_size, 1)).to(self.device)
					true_labels = torch.normal(0.9, .1, size=(batch_size, 1)).clamp(0.8, 1.).to(
						self.device)  # soft target labels
					# train the discriminator with real batch
					true_discriminator_out = self.discriminator(state, action)
					true_discriminator_loss = self.gan_loss(true_discriminator_out, true_labels)
					true_discriminator_loss.backward()

					# train the discriminator with fake batch
					noise = self.generator.generate_noise(batch_size).to(self.device)
					generator_actions = self.generator(state, noise)
					# false_labels = torch.zeros((batch_size, 1)).to(self.device)
					false_labels = torch.normal(0.1, .1, size=(batch_size, 1)).clamp(0., 0.2).to(
						self.device)  # soft target labels
					gen_for_dis_out = self.discriminator(state, generator_actions.detach())
					gen_dis_loss = self.gan_loss(gen_for_dis_out, false_labels)
					gen_dis_loss.backward()
					discriminator_loss = true_discriminator_loss + gen_dis_loss
					self.discriminator_optimizer.step()

					# train generator (invert the labels here as this can help with training)
					self.generator_optimizer.zero_grad()
					gen_dis_out = self.discriminator(state, generator_actions)
					generator_loss = self.gan_loss(gen_dis_out, true_labels)
					generator_loss.backward()
					self.generator_optimizer.step()

					debug_gen_loss += generator_loss.item()
					debug_dis_loss += discriminator_loss.item()
					debug_dis_t_loss += true_discriminator_loss.item()
					debug_dis_f_loss += gen_dis_loss.item()
					debug_d_output_list.append(gen_for_dis_out[0].cpu().data.numpy())

				# Critic Training
				with torch.no_grad():
					# Duplicate next state 10 times
					duplicate_value = 10
					next_state = torch.repeat_interleave(next_state, duplicate_value, 0)

					# Compute value of perturbed actions sampled from the Generator
					noise = self.generator.generate_noise(duplicate_value * batch_size).to(self.device)
					next_actions = self.generator(next_state, noise)
					target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, next_actions))

					# Soft Clipped Double Q-learning
					# AWAC paper says just taking min is fine, may want to try that here
					target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(
						target_Q1, target_Q2)

					# Take max over each action sampled from the Generator
					target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
					target_Q = reward + not_done * self.discount * target_Q

				current_Q1, current_Q2 = self.critic(state, action)
				critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				self.critic_optimizer.step()

				# Pertubation Model / Action Training
				noise = self.generator.generate_noise(batch_size).to(self.device)
				sampled_actions = self.generator(state, noise)
				perturbed_actions = self.actor(state, sampled_actions)

				# Update through DPG
				actor_loss = -self.critic.q1(state, perturbed_actions).mean()

				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				debug_critic_loss += critic_loss.item()
				debug_actor_loss += actor_loss.item()
				debug_sampled_actions_list.append(sampled_actions[0].cpu().data.numpy())
				debug_next_actions_list.append(next_actions[0].cpu().data.numpy())

				# Update Target Networks for A and C
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			else:
				# combined G and D model training
				if self.use_wgan:
					noise = self.generator.generate_noise(batch_size).to(self.device)
					generator_actions, a_actions = self.generator(state, noise)
					a_actions.detach()
					# train discriminator real
					self.discriminator_optimizer.zero_grad()
					true_discriminator_out, d_critic_1, d_critic_2 = self.discriminator(state, action)
					true_discriminator_loss = -torch.mean(true_discriminator_out)
					# train discriminator fake
					gen_for_dis_out, d_critic_1, d_critic_2 = self.discriminator(state, generator_actions) # not sure if i want to detach here
					d_critic_1.detach(), d_critic_2.detach()
					gen_dis_loss = torch.mean(gen_for_dis_out)
					# gradient penalty
					alpha = torch.rand((batch_size, self.action_dim)).to(self.device)
					x_hat = alpha * action.data + (1. - alpha) * generator_actions.data
					x_hat.requires_grad = True
					pred_hat, d_critic_1, d_critic_2 = self.discriminator(state, x_hat)
					d_critic_1.detach(), d_critic_2.detach()
					gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
									 create_graph=True, retain_graph=True, only_inputs=True)[0]
					gradient_penalty = self.gp_lambda * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

					discriminator_loss = true_discriminator_loss + gen_dis_loss + gradient_penalty
					discriminator_loss.backward()
					self.discriminator_optimizer.step()
					# debugging
					debug_dis_loss += discriminator_loss.item()
					debug_dis_t_loss += true_discriminator_loss.item()
					debug_dis_f_loss += gen_dis_loss.item()
					debug_dis_gp_loss += gradient_penalty.item()

					if (it+1) % self.wgan_d_iter == 0:
						# train generator
						self.generator_optimizer.zero_grad()
						generator_actions, a_actions = self.generator(state, noise) # only one batch of noise done above
						a_actions.detach()
						gen_dis_out, d_critic_1, d_critic_2 = self.discriminator(state, generator_actions)
						d_critic_1.detach(), d_critic_2.detach()
						generator_loss = -torch.mean(gen_dis_out)
						generator_loss.backward()
						self.generator_optimizer.step()
						debug_gen_loss += generator_loss.item()
						debug_d_output_list.append(gen_dis_out[0].cpu().data.numpy())
				else:
					# alternate way to train D and G, train D first, backwards on each part then do G
					self.discriminator_optimizer.zero_grad()
					true_labels = torch.normal(0.9, .1, size=(batch_size, 1)).clamp(0.8, 1.).to(self.device)  # soft target labels
					# train the discriminator with real batch
					true_discriminator_out, d_critic_1, d_critic_2 = self.discriminator(state, action)
					d_critic_1.detach(), d_critic_2.detach()
					true_discriminator_loss = self.gan_loss(true_discriminator_out, true_labels)
					true_discriminator_loss.backward()

					# train the discriminator with fake batch
					noise = self.generator.generate_noise(batch_size).to(self.device)
					generator_actions, a_actions = self.generator(state, noise)
					a_actions.detach()
					false_labels = torch.normal(0.1, .1, size=(batch_size, 1)).clamp(0., 0.2).to(
						self.device)  # soft target labels
					gen_for_dis_out, d_critic_1, d_critic_2 = self.discriminator(state, generator_actions.detach())
					d_critic_1.detach(), d_critic_2.detach()
					gen_dis_loss = self.gan_loss(gen_for_dis_out, false_labels)
					gen_dis_loss.backward()
					discriminator_loss = true_discriminator_loss + gen_dis_loss
					self.discriminator_optimizer.step()

					# train generator (invert the labels here as this can help with training)
					self.generator_optimizer.zero_grad()
					gen_dis_out, d_critic_1, d_critic_2 = self.discriminator(state, generator_actions)
					d_critic_1.detach(), d_critic_2.detach()
					generator_loss = self.gan_loss(gen_dis_out, true_labels)
					generator_loss.backward()
					self.generator_optimizer.step()

					# debugging
					debug_gen_loss += generator_loss.item()
					debug_dis_loss += discriminator_loss.item()
					debug_dis_t_loss += true_discriminator_loss.item()
					debug_dis_f_loss += gen_dis_loss.item()
					debug_d_output_list.append(gen_dis_out[0].cpu().data.numpy())


				# Critic Training
				with torch.no_grad():
					# Duplicate next state 10 times
					duplicate_value = 10
					next_state = torch.repeat_interleave(next_state, duplicate_value, 0)

					# Compute value of perturbed actions sampled from the Generator
					noise = self.generator.generate_noise(duplicate_value*batch_size).to(self.device)
					next_actions, _ = self.generator(next_state, noise)
					# generated actions then fed into actor target for perturbed actions
					_, a = self.generator_target(next_state, next_actions)
					_, target_Q1, target_Q2 = self.discriminator_target(next_state, a)

					# Soft Clipped Double Q-learning
					# AWAC paper says just taking min is fine, may want to try that here
					target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)

					# Take max over each action sampled from the Generator
					target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
					target_Q = reward + not_done * self.discount * target_Q

				d_discrim_out, current_Q1, current_Q2 = self.discriminator(state, action)
				d_discrim_out.detach()
				critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

				self.discriminator_optimizer.zero_grad()
				critic_loss.backward()
				self.discriminator_optimizer.step()

				# Pertubation Model / Action Training
				# sampled_actions = self.vae.decode(state)
				noise = self.generator.generate_noise(batch_size).to(self.device)
				sampled_actions, perturbed_actions = self.generator(state, noise)
				sampled_actions.detach()
				# Update through DPG
				actor_loss = -self.discriminator.critic_vf(state, perturbed_actions).mean()

				self.generator_optimizer.zero_grad()
				actor_loss.backward()
				self.generator_optimizer.step()

				debug_critic_loss += critic_loss.item()
				debug_actor_loss += actor_loss.item()
				debug_sampled_actions_list.append(sampled_actions[0].cpu().data.numpy())
				debug_next_actions_list.append(next_actions[0].cpu().data.numpy())

				# Update Target Networks
				for param, target_param in zip(self.discriminator.parameters(), self.discriminator_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.generator.parameters(), self.generator_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			if not self.use_max_a:
				# train RND on the states and actions
				self.rnd_optimizer.zero_grad()
				pred_feature, tgt_feature = self.rnd(state, action)
				rnd_loss = self.rnd_loss(pred_feature, tgt_feature.detach())
				rnd_loss.backward()
				self.rnd_optimizer.step()
				rnd_differences = (pred_feature - tgt_feature).pow(2).sum(dim=1).reshape(-1, 1).cpu().data.numpy().flatten()
				self.rnd_percentile_list.append(np.percentile(rnd_differences, self.rnd_percentile))

		if not self.use_max_a:
			self.rnd_cutoff_value = np.percentile(self.rnd_percentile_list, 50) # put on device at action select time

		# print out some stuff for debugging
		if True:
			print("g/d loss is ", np.round(debug_gen_loss/iterations, 3), np.round(debug_dis_loss/iterations, 3),
				  np.round(debug_dis_t_loss/iterations, 3), np.round(debug_dis_f_loss/iterations, 3), np.round(debug_dis_gp_loss/iterations, 3))
			print("a/c loss is ", np.round(debug_actor_loss/iterations, 2), np.round(debug_critic_loss / iterations, 2))
			print("sampled action median: ", np.median(debug_sampled_actions_list, axis=0))
			print("sampled action std: ", np.std(debug_sampled_actions_list, axis=0))
			print("next action median: ", np.median(debug_next_actions_list, axis=0))
			print("next action std: ", np.std(debug_next_actions_list, axis=0))
			print("D output mean: ", np.mean(debug_d_output_list, axis=0))
			print("D output std: ", np.std(debug_d_output_list, axis=0))