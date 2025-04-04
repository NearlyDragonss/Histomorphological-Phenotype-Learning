from models.ops import *
import tensorflow as tf
from torch import nn

display = False #todo change back

class EncoderResnetContrastive(torch.nn.Module):
	def __init__(self, model_name, z_dim, h_dim, layers, spectral, activation, init='xavier',
				 regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
		super(EncoderResnetContrastive, self).__init__()
		self.model_name = model_name
		self.z_dim = z_dim
		self.h_dim = h_dim
		self.layers = layers
		self.spectral = spectral
		self.activation = activation
		self.init = init
		self.regularizer = regularizer
		self.normalization = normalization
		self.attention = attention
		self.down = down
		self.name = name

	def forward(self, model, images, is_train):
		if '_0' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive(model, images, is_train)
		elif '_1' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive_1(model, images, is_train)
		elif '_2' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive_2(model, images, is_train)
		elif '_3' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive_3(model, images, is_train)
		elif '_4' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive_4(model, images, is_train)
		elif '_5' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive_5(model, images, is_train)
		elif '_6' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive_6(model, images, is_train)
		elif '_7' in self.model_name:
			conv_space, h, z = self.encoder_resnet_contrastive_7(model, images, is_train)
		return conv_space, h, z


	def encoder_resnet_contrastive_6(self, model,  images, is_train):
		channels = [64, 128, 256, 512, 1024, 2048]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()

		for layer in range(self.layers):
			# ResBlock.
			net = residual_block(model=model, inputs=images, filter_size=3, stride=1, padding='SAME', scope=layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer, activation=self.activation)
			# Attention layer.
			if self.attention is not None and net.shape.as_list()[1]==self.attention:
				net = attention_block_2(model=model, x=net, spectral=True, init=self.init, regularizer=self.regularizer, scope=self.layers)

			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME',
								conv_type=self.down, spectral=self.spectral, init=self.init, regularizer=self.regularizer, scope=layer)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(pool_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten()
		conv_space = flatten(conv_space)

		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init, regularizer=self.regularizer,
					scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer, scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer, scope='z_rep')

		print()
		return conv_space, h, z


	def encoder_resnet_contrastive_2(self, model, images, is_train):
		channels = [32, 64, 128, 256, 512, 1024]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()

		net = convolutional(model=model, inputs=images, output_channels=16, filter_size=7, stride=1, padding='SAME',
							conv_type='convolutional', spectral=self.spectral, init=self.init, regularizer=self.regularizer,
							scope='intital_layer', display=True)

		for layer in range(self.layers):
			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sa' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer,
								 activation=self.activation)
			# Attention layer.
			if self.attention is not None and net.shape.as_list()[1]==self.attention:
				net = attention_block_2(model=model, x=net, spectral=True, init=self.init, regularizer=self.regularizer, scope=self.layers)

			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sb' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer, activation=self.activation)

			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2,
								padding='SAME', conv_type=self.down, spectral=self.spectral, init=self.init,
								regularizer=self.regularizer, scope=layer)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(pool_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten()
		conv_space = flatten(conv_space)

		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer,
					scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer,
				  scope='z_rep')

		print()
		return conv_space, h, z


	def encoder_resnet_contrastive_1(self, model, images, is_train):
		channels = [32, 64, 128, 256, 512, 1024]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()


		for layer in range(self.layers):
			# ResBlock.
			net = residual_block(model=model, inputs=images, filter_size=5, stride=1, padding='SAME', scope='%sa' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer, activation=self.activation)
			# Attention layer.
			if self.attention is not None and net.shape.as_list()[1]==self.attention:
				net = attention_block_2(model=model, x=net, spectral=True, init=self.init, regularizer=self.regularizer, scope=self.layers)

			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sb' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer, activation=self.activation)

			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2,
								padding='SAME', conv_type=self.down, spectral=self.spectral, init=self.init,
								regularizer=self.regularizer, scope=layer)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(pool_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten() # todo: check is correct
		conv_space = flatten(conv_space)

		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer, scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer, scope='z_rep')

		print()
		return conv_space, h, z


	def encoder_resnet_contrastive_3(self, model, images, is_train):

		channels = [32, 64, 128, 256, 512, 1024]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()
		net = convolutional(model=model, inputs=images, output_channels=32, filter_size=7, stride=2, padding='SAME',
							conv_type='convolutional', spectral=self.spectral, init=self.init,
							regularizer=self.regularizer, scope='intital_layer', display=True)
		for layer in range(self.layers):
			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sa' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer,
								 activation=self.activation)
			# Attention layer.
			if self.attention is not None and net.shape[2]==self.attention:
				net = attention_block_2(model=model, x=net, spectral=True, init=self.init, regularizer=self.regularizer, scope=self.layers)

			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=5, stride=1, padding='SAME', scope='%sb' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer, activation=self.activation)

			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2,
								padding='SAME', conv_type=self.down, spectral=self.spectral, init=self.init,
								regularizer=self.regularizer, scope=layer)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten() # todo: check is correct
		conv_space = flatten(conv_space)

		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(model=model, inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(model=model, inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(model=model, inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init,
				  regularizer=self.regularizer, scope='z_rep')

		print()
		return conv_space, h, z


	def encoder_resnet_contrastive_4(self, model, images, is_train):
		channels = [32, 64, 128, 256, 512, 1024]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()

		net = convolutional(model=model, inputs=images, output_channels=32, filter_size=7, stride=2, padding='SAME',
							conv_type='convolutional', spectral=self.spectral, init=self.init,
							regularizer=self.regularizer, scope='intital_layer', display=True)

		for layer in range(self.layers):
			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer,
								 activation=self.activation)
			# Attention layer.
			if self.attention is not None and net.shape.as_list()[1]==self.attention:
				net = attention_block_2(model=model, x=net, spectral=True, init=self.init, regularizer=self.regularizer,
										scope=self.layers)

			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2,
								padding='SAME', conv_type=self.down, spectral=self.spectral, init=self.init,
								regularizer=self.regularizer, scope=layer)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(pool_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten() # todo: check is correct
		conv_space = flatten(conv_space)

		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer,
					scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer,
				  scope='z_rep')

		print()
		return conv_space, h, z


	def encoder_resnet_contrastive_5(self, model, images, is_train):

		channels = [32, 64, 128, 256, 512, 1024]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()

		net = convolutional(model=model, inputs=images, output_channels=32, filter_size=7, stride=2, padding='SAME',
							conv_type='convolutional', spectral=self.spectral, init=self.init,
							regularizer=self.regularizer, scope='intital_layer', display=True)

		for layer in range(self.layers):
			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=3, stride=1, padding='SAME', scope='%sa' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer,
								 activation=self.activation)
			# Attention layer.
			if self.attention is not None and net.shape.as_list()[1]==self.attention:
				net = attention_block_2(model=self.model, x=net, spectral=True, init=self.init, regularizer=self.regularizer,
										scope=self.layers)
			# ResBlock.
			net = residual_block(model=model, inputs=net, filter_size=3, stride=1, padding='SAME', scope='%sb' % layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer,
								 activation=self.activation)


			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME',
								conv_type=self.down, spectral=self.spectral, init=self.init,
								regularizer=self.regularizer, scope=layer)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(pool_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten() # todo: check is correct
		conv_space = flatten(conv_space)

		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init,
				  regularizer=self.regularizer, scope='z_rep')

		print()
		return conv_space, h, z


	def encoder_resnet_contrastive(self, model, images, is_train):
		channels = [32, 64, 128, 256, 512, 1024]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()

		for layer in range(self.layers):
			# ResBlock.
			net = residual_block(model=model, inputs=images, filter_size=3, stride=1, padding='SAME', scope=layer,
								 is_training=is_train, normalization=self.normalization, use_bias=True,
								 spectral=self.spectral, init=self.init, regularizer=self.regularizer,
								 activation=self.activation)
			# Attention layer.
			if self.attention is not None and net.shape.as_list()[1]==self.attention:
				net = attention_block_2(model=model, x=net, spectral=True, init=self.init, regularizer=self.regularizer,
										scope=self.layers)

			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2,
								padding='SAME', conv_type=self.down, spectral=self.spectral, init=self.init,
								regularizer=self.regularizer, scope=layer)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(pool_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten() # todo: check is correct
		conv_space = flatten(conv_space)

		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer,
				  scope='z_rep')

		print()
		return conv_space, h, z



	def encoder_resnet_contrastive_7(self, model, images, is_train):
		representation = list()
		channels = [32, 64, 128, 256, 512, 1024]
		if display:
			print('CONTRASTIVE ENCODER INFORMATION:')
			print('Channels: ', channels[:self.layers])
			print('Normalization: ', self.normalization)
			print('Activation: ', self.activation)
			print('Attention:  ', self.attention)
			print()

		for layer in range(self.layers):
			# ResBlock.
			net, style_resnet = residual_block(model=model, inputs=images, filter_size=3, stride=1, padding='SAME', scope=layer,
											   is_training=is_train, normalization=self.normalization,
											   use_bias=True, spectral=self.spectral, init=self.init,
											   regularizer=self.regularizer, activation=self.activation,
											   latent_dim=self.z_dim, style_extract_f=True)
			representation.append(style_resnet)

			# Attention layer.
			if self.attention is not None and net.shape.as_list()[1]==self.attention:
				net = attention_block_2(model=model, x=net, spectral=True, init=self.init, regularizer=self.regularizer, scope=self.layers)

			# Down.
			net = convolutional(model=model, inputs=net, output_channels=channels[layer], filter_size=4, stride=2,
								padding='SAME', conv_type=self.down, spectral=self.spectral, init=self.init,
								regularizer=self.regularizer, scope=layer)
			# Style extraction.
			style_conv = style_extract_2(inputs=net, latent_dim=self.z_dim, spectral=self.spectral, init=self.init,
										 regularizer=self.regularizer, scope=layer)
			representation.append(style_conv)
			if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
			net = self.activation(net)

		# Feature space extraction
		max_pool = nn.MaxPool2d(pool_size=2, stride=2)
		conv_space = max_pool(net)
		flatten = nn.Flatten() # todo: check is correct
		conv_space = flatten(conv_space)


		# Flatten.
		flatten = nn.Flatten()
		net = flatten(net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=self.spectral, init=self.init,
					regularizer=self.regularizer, scope='h_rep')
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		h = self.activation(net)

		net = dense(inputs=h, out_dim=self.h_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer,
					scope=2)
		if self.normalization is not None: net = self.normalization(inputs=net, training=is_train)
		net = self.activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=self.z_dim, spectral=self.spectral, init=self.init, regularizer=self.regularizer,
				  scope='z_rep')
		representation.append(z)

		representation = tf.concat(representation, axis=1)

		print()
		return h, z, representation



def encoder_resnet_contrastive_SimSiam(images, z_dim, h_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init,
								 regularizer=regularizer, activation=activation)
			# Attention layer.
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)

			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)

		# Feature space extraction
		conv_space = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=[2,2])
		conv_space = tf.layers.flatten(inputs=conv_space)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope='h_rep')
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		h = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)
		if normalization is not None: h = normalization(inputs=h, training=is_train)
		net = activation(h)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')

	print()
	return conv_space, h, z


def encoder_resnet_contrastive_SwAV(images, z_dim, prototype_dim, layers, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, attention=None, down='downscale', name='contrastive_encoder'):
	net = images
	channels = [32, 64, 128, 256, 512, 1024]
	if display:
		print('CONTRASTIVE ENCODER INFORMATION:')
		print('Channels: ', channels[:layers])
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print('Attention:  ', attention)
		print()

	with tf.variable_scope(name, reuse=reuse):

		for layer in range(layers):
			# ResBlock.
			net = residual_block(inputs=net, filter_size=3, stride=1, padding='SAME', scope=layer, is_training=is_train, normalization=normalization, use_bias=True, spectral=spectral, init=init,
								 regularizer=regularizer, activation=activation)
			# Attention layer.
			if attention is not None and net.shape.as_list()[1]==attention:
				net = attention_block_2(net, spectral=True, init=init, regularizer=regularizer, scope=layers)

			# Down.
			net = convolutional(inputs=net, output_channels=channels[layer], filter_size=4, stride=2, padding='SAME', conv_type=down, spectral=spectral, init=init, regularizer=regularizer, scope=layer)
			if normalization is not None: net = normalization(inputs=net, training=is_train)
			net = activation(net)

		# Flatten.
		net = tf.layers.flatten(inputs=net)

		# H Representation Layer.
		net = dense(inputs=net, out_dim=channels[-1], spectral=spectral, init=init, regularizer=regularizer, scope=1)
		# if normalization is not None: net = normalization(inputs=net, training=is_train)
		h = activation(net)

		net = dense(inputs=h, out_dim=int(channels[-1]/2), spectral=spectral, init=init, regularizer=regularizer, scope=2)
		# if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Z Representation Layer.
		z = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='z_rep')

		# SwAV paper: Xnt is mapped to a vector representation by a non-linear mapping. Later projected ot a unit sphere.
		z_norm = tf.math.l2_normalize(z, axis=1, name='projection')

		prototype = dense(inputs=z_norm, out_dim=prototype_dim, use_bias=False, spectral=False, init=init, regularizer=regularizer, scope='prototypes')

	print()
	return h, z, z_norm, prototype


def byol_predictor(z_rep, z_dim, h_dim, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, name='encoder_predictor'):
	net = z_rep
	if display:
		print('PREDICTOR ENCODER INFORMATION:')
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	with tf.variable_scope(name, reuse=reuse):

		net = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		# Q Prediction.
		q_pred = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='q_pred')

	print()
	return q_pred


def dino_head(z_rep, z_dim, h_dim, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, name='encoder_predictor'):
	net = z_rep
	if display:
		print('PREDICTOR ENCODER INFORMATION:')
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	with tf.variable_scope(name, reuse=reuse):

		net = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=2)
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		net = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=1)
		if normalization is not None: net = normalization(inputs=net, training=is_train)
		net = activation(net)

		net = tf.math.l2_normalize(net, axis=1, name='projection')

		# Q Prediction.
		q_pred = dense(inputs=net, out_dim=z_dim, spectral=spectral, init=init, regularizer=regularizer, scope='q_pred', use_bias=False)

	print()
	return q_pred



def relational_module(aggregated_representations, h_dim, spectral, activation, is_train, reuse, init='xavier', regularizer=None, normalization=None, name='relational_module'):
	net = aggregated_representations
	if display:
		print('RELATIONAL REASONING MODULE INFORMATION:')
		print('Normalization: ', normalization)
		print('Activation: ', activation)
		print()

	with tf.variable_scope(name, reuse=reuse):
		net = dense(inputs=net, out_dim=h_dim, spectral=spectral, init=init, regularizer=regularizer, scope=1)
		if normalization is not None:
			net = normalization(inputs=net, training=is_train)
		net = activation(net)
		logits = dense(inputs=net, out_dim=1, spectral=spectral, init=init, regularizer=regularizer, scope=2)

	return logits


