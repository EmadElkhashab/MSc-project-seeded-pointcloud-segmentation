import numpy
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
from metric_loss_ops import triplet_semihard_loss

#import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from point_transformer_pytorch import PointTransformerLayer


def save_checkpoint(epoch, model, optimizer, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

class BatchNorm1d_P(nn.BatchNorm1d):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		#  input: (b, n, c)
		return super().forward(x.transpose(1, 2)).transpose(1, 2)

class PointTransformerBlock(nn.Module):
	def __init__(self, in_channels, out_channels, pos_mlp_hidden_dim=32, attn_mlp_hidden_mult=4, num_neighbors=16):
		super(PointTransformerBlock, self).__init__()
		self.fc1 = nn.Linear(in_channels, out_channels)
		self.bn1 = BatchNorm1d_P(out_channels)
		self.point_transformer_layer = PointTransformerLayer(
			dim=out_channels,
			pos_mlp_hidden_dim=pos_mlp_hidden_dim,
			attn_mlp_hidden_mult=attn_mlp_hidden_mult,
			num_neighbors=num_neighbors 
		)
		self.bn2 = BatchNorm1d_P(out_channels)
		self.fc2 = nn.Linear(out_channels, out_channels)
		self.bn3 = BatchNorm1d_P(out_channels)
		self.relu = nn.ReLU(inplace=True)


	def forward(self, x, p):
		# x shape: (batch_size, seq_len, num_points, in_channels)
		batch_size, num_points, in_channels = x.shape
		#p = p.view(batch_size * seq_len, num_points, 3)
		#x = x.view(batch_size * seq_len, num_points, in_channels)
		
		x = self.relu(self.bn1(self.fc1(x)))
		residual = x
		mask = torch.ones(batch_size, num_points).bool().to(x.device)
		x = self.relu(self.bn2(self.point_transformer_layer(x, p, mask=mask)))
		y = self.relu(self.bn3(self.fc2(x + residual)))
		#y = y.view(batch_size, seq_len, num_points, -1)
		#p = p.view(batch_size, seq_len, num_points, -1)
		return y, p


class RegionTransformer(nn.Module):
	def __init__(self, batch_size=32,
				 seq_len=1, # Number of region growing steps unrolled
				 num_inlier_points=512,
				 num_neighbour_points=512,
				 feature_dim=10):
		super(RegionTransformer, self).__init__()
		self.num_points = num_inlier_points
		self.B1 = [64, 64]
		self.B2 = [64, 128, 128]
		self.B3 = [128*2+64, 128]
		
		# Do I include this in the PointTransformer blocks?
		# Initial up-dimension layers for inlier and neighbour points (to be updated)
		self.fc1 = nn.Linear(feature_dim-3, self.B1[0])
		self.neigbour_fc1 = nn.Linear(feature_dim-3, self.B1[0])

		# Point Transformer Layers for inlier set
		for i in range(len(self.B1)-1):
			inlier_layer =  PointTransformerBlock(in_channels=self.B1[i], out_channels=self.B1[i+1])
			neighbour_layer = PointTransformerBlock(in_channels=self.B1[i],out_channels=self.B1[i+1])
			######### Need a diagram drawn with dimensions #########
			# inlier and linear have different input shapes
			setattr(self, 'B1_inlier_' + str(i), inlier_layer)
			setattr(self, 'B1_neighbour_' + str(i), neighbour_layer)
		
		# Point Transformer Layers for outlier set
		for i in range(len(self.B2)-1):
			inlier_layer =  PointTransformerBlock(in_channels=self.B2[i], out_channels=self.B2[i+1])
			neighbour_layer = PointTransformerBlock(in_channels=self.B2[i],out_channels=self.B2[i+1])
			setattr(self, 'B2_inlier_' + str(i), inlier_layer)
			setattr(self, 'B2_neighbour_' + str(i), neighbour_layer)

		# Global Average Pooling
		###### Does this work as intended ######
		# Do I reduce along the num_points dimension?
		# https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html
		self.max_pool = torch.nn.AdaptiveMaxPool2d((1, None))
		self.max_pool_neigh = torch.nn.AdaptiveMaxPool2d((1, None))
		
		# B3
		for i in range(len(self.B3)-1):
			inlier_layer =  PointTransformerBlock(in_channels=self.B3[i], out_channels=self.B3[i+1])
			neighbour_layer = PointTransformerBlock(in_channels=self.B3[i],out_channels=self.B3[i+1])
			setattr(self, 'B3_inlier_' + str(i), inlier_layer)
			setattr(self, 'B3_neighbour_' + str(i), neighbour_layer)
		
		# Final classification layer
		self.remove_mask = nn.Linear(self.B3[-1], 1)  
		self.add_mask = nn.Linear(self.B3[-1], 1)  


	def forward(self, inlier_points, neighbour_points):
		inlier_points.unsqueeze(1)
		neighbour_points.unsqueeze(1)

		p = inlier_points[:,:,:3]
		p_neigh = neighbour_points[:,:,:3]
		x = inlier_points[:,:,3:] 
		x_neigh = neighbour_points[:,:,3:]
		
		x = self.fc1(x)
		x_neigh = self.neigbour_fc1(x_neigh)

		for i in range(len(self.B1)-1):
			inlier_layer = getattr(self, 'B1_inlier_' + str(i))
			neighbour_layer = getattr(self, 'B1_neighbour_' + str(i))
			x, p = inlier_layer(x, p)
			x_neigh, p_neigh = neighbour_layer(x_neigh, p_neigh)

		residual = x
		residual_neigh = x_neigh

		for i in range(len(self.B2)-1):
			inlier_layer = getattr(self, 'B2_inlier_' + str(i))
			neighbour_layer = getattr(self, 'B2_neighbour_' + str(i))
			x, p = inlier_layer(x, p)
			x_neigh, p_neigh  = neighbour_layer(x_neigh, p_neigh)

		# Global Max Pooling
		x = self.max_pool(x)
		x_neigh = self.max_pool(x_neigh)
		
		# concatenate x and x_neigh along feature dimension
		x = torch.cat([x, x_neigh], dim=-1)
		x_broadcasted = torch.broadcast_to(x, (-1, self.num_points, -1))

		x = torch.cat([x_broadcasted, residual], dim=-1)
		x_neigh = torch.cat([x_broadcasted, residual_neigh], dim=-1)

		# B3
		for i in range(len(self.B3)-1):
			inlier_layer = getattr(self, 'B3_inlier_' + str(i))
			neighbour_layer = getattr(self, 'B3_neighbour_' + str(i))
			x, p = inlier_layer(x, p)
			x_neigh, p_neigh = neighbour_layer(x_neigh, p_neigh)

		# Final classification
		remove_mask_logits = self.remove_mask(x).squeeze(dim=2)
		add_mask_logits = self.add_mask(x_neigh).squeeze(dim=2)

		return remove_mask_logits, add_mask_logits


#	Disable eager exec because tf.placeholder is not permitted in eager execution.
tf.compat.v1.disable_eager_execution()

def loadFromH5(filename, load_labels=True):
	f = h5py.File(filename,'r')
	all_points = f['points'][:]
	count_room = f['count_room'][:]
	tmp_points = []
	idp = 0
	for i in range(len(count_room)):
		tmp_points.append(all_points[idp:idp+count_room[i], :])
		idp += count_room[i]
	f.close()
	room = []
	labels = []
	class_labels = []
	if load_labels:
		for i in range(len(tmp_points)):
			room.append(tmp_points[i][:,:-2])
			labels.append(tmp_points[i][:,-2].astype(int))
			class_labels.append(tmp_points[i][:,-1].astype(int))
		return room, labels, class_labels
	else:
		return tmp_points

def savePCD(filename,points):
	if len(points)==0:
		return
	f = open(filename,"w")
	l = len(points)
	header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH %d
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS %d
DATA ascii
""" % (l,l)
	f.write(header)
	for p in points:
		rgb = (int(p[3]) << 16) | (int(p[4]) << 8) | int(p[5])
		f.write("%f %f %f %d\n"%(p[0],p[1],p[2],rgb))
	f.close()
	print('Saved %d points to %s' % (l,filename))

def savePLY(filename, points):
	f = open(filename,'w')
	f.write("""ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""" % len(points))
	for p in points:
		f.write("%f %f %f %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()
	print('Saved to %s: (%d points)'%(filename, len(points)))



class LrgNet:
	def __init__(self,batch_size, seq_len, num_inlier_points, num_neighbor_points, feature_size, lite=0):

		# with strat.scope():
		#	Double the number of channels to see if performance improves.
		if lite==0 or lite is None:
			#	Original arch.
			# CONV_CHANNELS = [64,64,64,128,512]
			# CONV2_CHANNELS = [256, 128]

			#	2x channels.
			# CONV_CHANNELS = [128, 128, 128, 256, 1024]
			# CONV2_CHANNELS = [512, 256]

			#	4x channels.
			CONV_CHANNELS = [256, 256, 256, 512, 2048]
			CONV2_CHANNELS = [1024, 512]

			#	8x channels.
			# CONV_CHANNELS = [512, 512, 512, 1024, 4096]
			# CONV2_CHANNELS = [2048, 1024]

			#	randy special
			# CONV_CHANNELS = [256, 256, 512, 512, 1024, 2048]
			# CONV2_CHANNELS = [1024, 512, 256]

			#	Deep.
			# CONV_CHANNELS = [128, 128, 128, 256, 256, 256, 512, 512, 512]
			# CONV2_CHANNELS = [256, 256, 128, 128]

			#	Deep with 2x channels.
			# CONV_CHANNELS = [256, 256, 256, 512, 512, 512, 1024, 1024, 1024]
			# CONV2_CHANNELS = [512, 512, 256, 256]
		elif lite==1:
			CONV_CHANNELS = [64,64]
			CONV2_CHANNELS = [64]
		elif lite==2:
			CONV_CHANNELS = [64,64,256]
			CONV2_CHANNELS = [64,64]
		self.kernel = [None]*len(CONV_CHANNELS)
		self.bias = [None]*len(CONV_CHANNELS)
		self.conv = [None]*len(CONV_CHANNELS)
		self.neighbor_kernel = [None]*len(CONV_CHANNELS)
		self.neighbor_bias = [None]*len(CONV_CHANNELS)
		self.neighbor_conv = [None]*len(CONV_CHANNELS)
		self.add_kernel = [None]*(len(CONV2_CHANNELS) + 1)
		self.add_bias = [None]*(len(CONV2_CHANNELS) + 1)
		self.add_conv = [None]*(len(CONV2_CHANNELS) + 1)
		self.remove_kernel = [None]*(len(CONV2_CHANNELS) + 1)
		self.remove_bias = [None]*(len(CONV2_CHANNELS) + 1)
		self.remove_conv = [None]*(len(CONV2_CHANNELS) + 1)
		self.inlier_tile = [None]*2
		self.neighbor_tile = [None]*2
		self.inlier_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size*seq_len, num_inlier_points, feature_size))
		self.neighbor_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size*seq_len, num_neighbor_points, feature_size))
		self.add_mask_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size*seq_len, num_neighbor_points))
		self.remove_mask_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size*seq_len, num_inlier_points))

		#CONVOLUTION LAYERS FOR INLIER SET
		for i in range(len(CONV_CHANNELS)):
			self.kernel[i] = tf.compat.v1.get_variable('lrg_kernel'+str(i), [1, feature_size if i==0 else CONV_CHANNELS[i-1], CONV_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.bias[i] = tf.compat.v1.get_variable('lrg_bias'+str(i), [CONV_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.nn.conv1d(input=self.inlier_pl if i==0 else self.conv[i-1], filters=self.kernel[i], stride=1, padding='VALID')
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])

			# noise = tf.random.normal(shape=tf.shape(self.conv[i]), mean=0.0, stddev=0.2, dtype=tf.float32)
			# self.conv[i] = self.conv[i] + noise

			self.conv[i] = tf.nn.relu(self.conv[i])

		#CONVOLUTION LAYERS FOR NEIGHBOR SET
		for i in range(len(CONV_CHANNELS)):
			self.neighbor_kernel[i] = tf.compat.v1.get_variable('lrg_neighbor_kernel'+str(i), [1, feature_size if i==0 else CONV_CHANNELS[i-1], CONV_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.neighbor_bias[i] = tf.compat.v1.get_variable('lrg_neighbor_bias'+str(i), [CONV_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.neighbor_conv[i] = tf.nn.conv1d(input=self.neighbor_pl if i==0 else self.neighbor_conv[i-1], filters=self.neighbor_kernel[i], stride=1, padding='VALID')
			self.neighbor_conv[i] = tf.nn.bias_add(self.neighbor_conv[i], self.neighbor_bias[i])

			# noise = tf.random.normal(shape=tf.shape(self.neighbor_conv[i]), mean=0.0, stddev=0.2, dtype=tf.float32)
			# self.neighbor_conv[i] = self.neighbor_conv[i] + noise
			
			self.neighbor_conv[i] = tf.nn.relu(self.neighbor_conv[i])

		#MAX POOLING
		self.pool = tf.reduce_max(input_tensor=self.conv[-1], axis=1)
		self.neighbor_pool = tf.reduce_max(input_tensor=self.neighbor_conv[-1], axis=1)
		self.combined_pool = tf.concat(axis=1, values=[self.pool, self.neighbor_pool])
		self.pooled_feature = self.combined_pool

		#CONCAT AFTER POOLING
		self.inlier_tile[0] = tf.tile(tf.reshape(self.pooled_feature,[batch_size*seq_len,-1,CONV_CHANNELS[-1]*2]) , [1,1,num_inlier_points])
		self.inlier_tile[0] = tf.reshape(self.inlier_tile[0],[batch_size*seq_len,num_inlier_points,-1])
		self.inlier_tile[1] = self.conv[1]
		self.inlier_concat = tf.concat(axis=2, values=self.inlier_tile)
		self.neighbor_tile[0] = tf.tile(tf.reshape(self.pooled_feature,[batch_size*seq_len,-1,CONV_CHANNELS[-1]*2]) , [1,1,num_neighbor_points])
		self.neighbor_tile[0] = tf.reshape(self.neighbor_tile[0],[batch_size*seq_len,num_neighbor_points,-1])
		self.neighbor_tile[1] = self.neighbor_conv[1]
		self.neighbor_concat = tf.concat(axis=2, values=self.neighbor_tile)

		#CONVOLUTION LAYERS AFTER POOLING
		for i in range(len(CONV2_CHANNELS)):
			self.add_kernel[i] = tf.compat.v1.get_variable('lrg_add_kernel'+str(i), [1, CONV_CHANNELS[-1]*2 + CONV_CHANNELS[1] if i==0 else CONV2_CHANNELS[i-1], CONV2_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.add_bias[i] = tf.compat.v1.get_variable('lrg_add_bias'+str(i), [CONV2_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.add_conv[i] = tf.nn.conv1d(input=self.neighbor_concat if i==0 else self.add_conv[i-1], filters=self.add_kernel[i], stride=1, padding='VALID')
			self.add_conv[i] = tf.nn.bias_add(self.add_conv[i], self.add_bias[i])
			
			# noise = tf.random.normal(shape=tf.shape(self.add_conv[i]), mean=0.0, stddev=0.5, dtype=tf.float32)
			# self.add_conv[i] = self.add_conv[i] + noise

			self.add_conv[i] = tf.nn.relu(self.add_conv[i])
		i += 1
		self.add_kernel[i] = tf.compat.v1.get_variable('lrg_add_kernel'+str(i), [1, CONV2_CHANNELS[-1], 2], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		self.add_bias[i] = tf.compat.v1.get_variable('lrg_add_bias'+str(i), [2], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		self.add_conv[i] = tf.nn.conv1d(input=self.add_conv[i-1], filters=self.add_kernel[i], stride=1, padding='VALID')
		self.add_conv[i] = tf.nn.bias_add(self.add_conv[i], self.add_bias[i])
		self.add_output = self.add_conv[i]

		for i in range(len(CONV2_CHANNELS)):
			self.remove_kernel[i] = tf.compat.v1.get_variable('lrg_remove_kernel'+str(i), [1, CONV_CHANNELS[-1]*2 + CONV_CHANNELS[1] if i==0 else CONV2_CHANNELS[i-1], CONV2_CHANNELS[i]], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
			self.remove_bias[i] = tf.compat.v1.get_variable('lrg_remove_bias'+str(i), [CONV2_CHANNELS[i]], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
			self.remove_conv[i] = tf.nn.conv1d(input=self.inlier_concat if i==0 else self.remove_conv[i-1], filters=self.remove_kernel[i], stride=1, padding='VALID')
			self.remove_conv[i] = tf.nn.bias_add(self.remove_conv[i], self.remove_bias[i])
			
			# noise = tf.random.normal(shape=tf.shape(self.remove_conv[i]), mean=0.0, stddev=0.5, dtype=tf.float32)
			# self.remove_conv[i] = self.remove_conv[i] + noise

			self.remove_conv[i] = tf.nn.relu(self.remove_conv[i])
		i += 1
		self.remove_kernel[i] = tf.compat.v1.get_variable('lrg_remove_kernel'+str(i), [1, CONV2_CHANNELS[-1], 2], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		self.remove_bias[i] = tf.compat.v1.get_variable('lrg_remove_bias'+str(i), [2], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		self.remove_conv[i] = tf.nn.conv1d(input=self.remove_conv[i-1], filters=self.remove_kernel[i], stride=1, padding='VALID')
		self.remove_conv[i] = tf.nn.bias_add(self.remove_conv[i], self.remove_bias[i])
		self.remove_output = self.remove_conv[i]

		#LOSS FUNCTIONS
		def weighted_cross_entropy(logit, label):
			pos_mask = tf.compat.v1.where(tf.cast(label, tf.bool))
			neg_mask = tf.compat.v1.where(tf.cast(1 - label, tf.bool))
			pos_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logit, pos_mask), labels=tf.gather_nd(label, pos_mask)))
			neg_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.gather_nd(logit, neg_mask), labels=tf.gather_nd(label, neg_mask)))
			pos_loss = tf.cond(pred=tf.math.is_nan(pos_loss), true_fn=lambda: 0.0, false_fn=lambda: pos_loss)
			neg_loss = tf.cond(pred=tf.math.is_nan(neg_loss), true_fn=lambda: 0.0, false_fn=lambda: neg_loss)
			return pos_loss + neg_loss

		self.add_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.add_output, labels=self.add_mask_pl))
		self.add_acc = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.argmax(input=self.add_output, axis=-1), tf.cast(self.add_mask_pl, dtype=tf.int64)), tf.float32))
		TP = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(tf.equal(tf.argmax(input=self.add_output, axis=-1), 1), tf.equal(self.add_mask_pl, 1)), tf.float32))
		self.add_prc = TP / (tf.cast(tf.reduce_sum(input_tensor=tf.argmax(input=self.add_output, axis=-1)), tf.float32) + 1)
		self.add_rcl = TP / (tf.cast(tf.reduce_sum(input_tensor=self.add_mask_pl), tf.float32) + 1)
		self.remove_loss = weighted_cross_entropy(self.remove_output, self.remove_mask_pl)
		self.remove_acc = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.argmax(input=self.remove_output, axis=-1), tf.cast(self.remove_mask_pl, dtype=tf.int64)), tf.float32))
		self.remove_mask = tf.nn.softmax(self.remove_output, axis=-1)[:, :, 1] > 0.5
		TP = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(self.remove_mask, tf.equal(self.remove_mask_pl, 1)), tf.float32))
		self.remove_prc = TP / (tf.reduce_sum(input_tensor=tf.cast(self.remove_mask, tf.float32)) + 1)
		self.remove_rcl = TP / (tf.cast(tf.reduce_sum(input_tensor=self.remove_mask_pl), tf.float32) + 1)

		self.loss = self.add_loss + self.remove_loss
		batch = tf.Variable(0)
		optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)

class MCPNet:
	def __init__(self,batch_size, neighbor_size, feature_size, hidden_size, embedding_size):
		self.input_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, feature_size-2))
		self.label_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))
		self.neighbor_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, neighbor_size, feature_size))

		#NETWORK_WEIGHTS
		kernel1 = tf.compat.v1.get_variable('mcp_kernel1', [1,feature_size,hidden_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias1 = tf.compat.v1.get_variable('mcp_bias1', [hidden_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		kernel2 = tf.compat.v1.get_variable('mcp_kernel2', [1,hidden_size,hidden_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias2 = tf.compat.v1.get_variable('mcp_bias2', [hidden_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		kernel3 = tf.compat.v1.get_variable('mcp_kernel3', [feature_size-2+hidden_size, hidden_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias3 = tf.compat.v1.get_variable('mcp_bias3', [hidden_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		kernel4 = tf.compat.v1.get_variable('mcp_kernel4', [hidden_size, embedding_size], initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
		bias4 = tf.compat.v1.get_variable('mcp_bias4', [embedding_size], initializer=tf.compat.v1.constant_initializer(0.0), dtype=tf.float32)
		self.kernels = [kernel1, kernel2, kernel3, kernel4]
		self.biases = [bias1, bias2, bias3, bias4]

		#MULTI-VIEW CONTEXT POOLING
		neighbor_fc = tf.nn.conv1d(input=self.neighbor_pl, filters=kernel1, stride=1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias1)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.nn.conv1d(input=neighbor_fc, filters=kernel2, stride=1, padding='VALID')
		neighbor_fc = tf.nn.bias_add(neighbor_fc, bias2)
		neighbor_fc = tf.nn.relu(neighbor_fc)
		neighbor_fc = tf.reduce_max(input_tensor=neighbor_fc, axis=1)
		concat = tf.concat(axis=1, values=[self.input_pl, neighbor_fc])

		#FEATURE EMBEDDING BRANCH (for instance label prediction)
		fc3 = tf.matmul(concat, kernel3)
		fc3 = tf.nn.bias_add(fc3, bias3)
		fc3 = tf.nn.relu(fc3)
		self.fc4 = tf.matmul(fc3, kernel4)
		self.fc4 = tf.nn.bias_add(self.fc4, bias4)
		self.embeddings = tf.nn.l2_normalize(self.fc4, axis=1)
		self.triplet_loss = triplet_semihard_loss(self.label_pl, self.embeddings)

		#LOSS FUNCTIONS
		self.loss = self.triplet_loss
		batch = tf.Variable(0)
		optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss, global_step=batch)
