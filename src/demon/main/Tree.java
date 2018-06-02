package demon.main;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class Tree {
	
	private TreeNode root;
	private int min_sample_split;
	private double min_child_weight;
	private int max_depth;
	private double colsample;
	private double rowsample;
	private double lambda;
	private double gamma;
	private int num_thread;
	
	public int nodes_cnt = 0;
	public int nan_nodes_cnt = 0;
	
	private Queue<TreeNode> alive_nodes = new LinkedList<>();
	
	public Tree(int min_sample_split,
				double min_child_weight,
				int max_depth,
				double colsample,
				double rowsample,
				double lambda,
				double gamma,
				int num_thread) {
		this.min_sample_split = min_sample_split;
		this.min_child_weight = min_child_weight;
		this.max_depth = max_depth;
		this.colsample = colsample;
		this.rowsample = rowsample;
		this.lambda = lambda;
		this.gamma = gamma;
		
		if (num_thread == -1) {
			this.num_thread = Runtime.getRuntime().availableProcessors();
		}
		else {
			this.num_thread = num_thread;
		}
		
		this.lambda = Math.max(this.lambda, 0.00001);
	}
	
	public TreeNode getRoot() {
		return this.root;
	}
	
	private double calculate_leaf_score(double G, double H) {
		return -G / (H + this.lambda);
	}
	
	private double[] calculate_split_gain(double G_left, double H_left, double G_nan, double H_nan, double G_total, double H_total) {
		double G_right = G_total - G_left - G_nan;
		double H_right = H_total - H_left - H_nan;
		
		double gain_1 = 0.5 * (
				Math.pow(G_left, 2) / (H_left + lambda)
					+ Math.pow(G_right, 2) / (H_right + lambda)
					+ Math.pow(G_nan, 2) / (H_nan + lambda)
					- Math.pow(G_total, 2) / (H_total + lambda)) - gamma;
		
		double gain_2 = 0.5 * (
                Math.pow(G_left+G_nan,2)/(H_left+H_nan+lambda)
                        + Math.pow(G_right,2)/(H_right+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda))-gamma;

        //if we let those with missing value go to right child
        double gain_3 = 0.5 * (
                Math.pow(G_left,2)/(H_left+lambda)
                        + Math.pow(G_right+G_nan,2)/(H_right+H_nan+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda))-gamma;
        
        double nan_go_to;
        double gain = Math.max(gain_1, Math.max(gain_2, gain_3));
        if (gain_1 == gain) {
        	nan_go_to = 0;
        }
        else if (gain_2 == gain) {
        	nan_go_to = 1;
        }
        else {
        	nan_go_to = 2;
        }
        
        // in this case, the trainset does not contains nan samples
        if (H_nan == 0 && G_nan == 0) {
        	nan_go_to = 3;
        }
        
        return new double[] {nan_go_to, gain};
	}
	
	public void fit(AttributeList attribute_list,
					ClassList class_list,
					RowSampler row_sampler,
					ColumnSampler col_sampler) {
		col_sampler.shuffle();
//		System.out.println(col_sampler.col_selected);
		row_sampler.shuffle();
		class_list.sampling(row_sampler.getRowMask());
		
		TreeNode root_node = new TreeNode(1, 1, attribute_list.getFeatureDim(), false);
		root_node.setGrad(sum(class_list.getGrad()));
		root_node.setHess(sum(class_list.getHess()));
		this.root = root_node;
		
		alive_nodes.offer(root_node);
		
		for (int i = 0; i < class_list.getDataSize(); ++i) {
			class_list.corresponding_tree_node[i] = root_node; // 每个 instance 对应的 tree root
		}
		
		// missing value processing
		class_list.update_grad_hess_missing_for_tree_node(attribute_list.getMissingValueAttributeList());
		build(attribute_list, class_list, col_sampler);
	}
	
	class ProcessEachNumericFeature implements Runnable{

		public int col;
		public AttributeList attribute_list;
		public ClassList class_list;
		
		public ProcessEachNumericFeature(int col, AttributeList attribute_list, ClassList class_list) {
			this.col = col;
			this.attribute_list = attribute_list;
			this.class_list = class_list;
		}
		
		@Override
		public void run() {
			// 优化的思路
			for (int interval = 0; interval < attribute_list.cutting_idx[col].length; ++interval) {
				int[] idxs = attribute_list.cutting_idx[col][interval]; // 分块的思想
				Set<TreeNode> nodes = new HashSet<>();
				for (int idx : idxs) { //累加每个样例 在 class list 中 存放的grad及hess
//					System.out.println(attribute_list.origin_feature[idx].X[col]);
//					System.out.println(attribute_list.cutting_thresholds[col][interval]);
					TreeNode treeNode = class_list.corresponding_tree_node[idx];  // 当前样例所在的树结点
					if (treeNode.isLeaf()) continue;
					nodes.add(treeNode);
					treeNode.getGLeft()[col] += class_list.getGrad()[idx];
					treeNode.getHLeft()[col] += class_list.getHess()[idx];
				}
				
				for (TreeNode node : nodes) {
					double G_left = node.getGLeft()[col];
					double H_left = node.getHLeft()[col];
					double G_total = node.grad;
					double H_total = node.hess;
					
					double G_nan = node.grad_missing[col];
					double H_nan = node.hess_missing[col];
					
					double[] ret = calculate_split_gain(G_left, H_left, G_nan, H_nan, G_total, H_total);
					double nan_go_to = ret[0];
					double gain = ret[1];
					
					node.update_best_split(col, attribute_list.cutting_thresholds[col][interval], gain, nan_go_to);
				}
			}
			
			
		}
		
	}
	
	private void build(AttributeList attribute_list, ClassList class_list, 
			ColumnSampler colSampler) {
		List<TreeNode> per = new ArrayList<>();
		while (!alive_nodes.isEmpty()) {
			nodes_cnt += alive_nodes.size();
			ExecutorService pool = Executors.newFixedThreadPool(num_thread);
			for (int col : colSampler.col_selected) {
				pool.execute(new ProcessEachNumericFeature(col, attribute_list, class_list)); //得到了best gain
			}
			
			pool.shutdown();
			
			try {
				pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			int cur_level_node_size = alive_nodes.size();
			Queue<TreeNode> new_tree_nodes = new LinkedList<>();
			
			for (int i = 0; i < cur_level_node_size; ++i) {
				TreeNode treeNode = alive_nodes.poll();
				per.add(treeNode);
				double[] best_col = treeNode.best_gains;
//				for (int j = 0; j < best_col.length; ++j) System.out.print(best_col[j] + " ");
//				System.out.println();
				List<Double> ret = treeNode.get_best_feature_threshold_gain();
				double best_feature = ret.get(0);
				double best_gain = ret.get(1);
				double best_nan_go_to = ret.get(2);
				double best_threshold = ret.get(3);
//				System.out.println(best_feature + " " + best_gain + " " + best_threshold);
				
				if (best_gain <= 0) {
					// this node is leaf node
					// best gain <= 0 why is leaf node?
					double leaf_score = calculate_leaf_score(treeNode.grad, treeNode.hess);
					treeNode.setLeafNode(leaf_score, true);
				}
				else {
					TreeNode leftChild = new TreeNode(3 * treeNode.index - 1, treeNode.depth + 1, treeNode.feature_dim, false);
					TreeNode rightChild = new TreeNode(3 * treeNode.index + 1, treeNode.depth + 1, treeNode.feature_dim, false);
					TreeNode nan_child = null;
					if (best_nan_go_to == 0) {
						nan_child = new TreeNode(3 * treeNode.index, treeNode.depth + 1, treeNode.feature_dim, false);
						nan_nodes_cnt += 1;
					}
					
					treeNode.setInternalNode(best_feature, best_threshold, best_nan_go_to, nan_child, leftChild, rightChild, false);
					
					new_tree_nodes.offer(leftChild);
					new_tree_nodes.offer(rightChild);
					
					if (nan_child != null) {
						new_tree_nodes.offer(nan_child);
					}
				}
			}
			
			//update class list.corresponding_tree_node
			class_list.update_corresponding_tree_node(attribute_list);  //27513 | 0
			class_list.update_Grad_Hess_numsample_for_tree_node();
			class_list.update_grad_hess_missing_for_tree_node(attribute_list.getMissingValueAttributeList());
			
			while (new_tree_nodes.size() != 0) {
				TreeNode treeNode = new_tree_nodes.poll();
				if (treeNode.depth >= max_depth || treeNode.hess < min_child_weight || 
						treeNode.num_sample <= min_sample_split) {
					treeNode.setLeafNode(calculate_leaf_score(treeNode.grad, treeNode.hess), true);
				}
				else {
					alive_nodes.offer(treeNode);
				}
			}
		}
	}
	
	/**
	 * For DMatrix Interface
	 * @param data
	 * @return
	 */
	public double[] predict(DMatrix data) {
		double[][] features = new double[data.getDataSize()][];
		for (int i = 0; i < data.getDataSize(); ++i) {
			features[i] = data.getOriginFeature()[i].X;
		}
		return predict(features);
	}
	
	public double[] predict(double[][] features) {
		ExecutorService pool = Executors.newFixedThreadPool(num_thread);
		List<Future> list = new ArrayList<>();
		for (int i = 0; i < features.length; ++i) {
			Callable c = new PredictCallable(features[i]);
			Future f = pool.submit(c);
			list.add(f);
		}
		
		// 组合关闭 线程池
		pool.shutdown();
		try {
			pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		double[] ret = new double[features.length];
		for (int i = 0; i < ret.length; ++i) {
			try {
				ret[i] = (double) list.get(i).get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		return ret;
	}
	
	class PredictCallable implements Callable{
		private double[] feature;
		public PredictCallable(double[] feature) {
			this.feature = feature;
		}
		
		@Override
		public Object call() throws Exception {
			TreeNode cur_tree_node = root; //直接调用root
			while (!cur_tree_node.isLeaf()) {
				if (DMatrix.isNAN(feature[cur_tree_node.split_feature])) {
					// it is missing value
					if (cur_tree_node.nan_go_to == 0) {
						cur_tree_node = cur_tree_node.nan_child;
					}
					else if (cur_tree_node.nan_go_to == 1) {
						cur_tree_node = cur_tree_node.left_child;
					}
					else if (cur_tree_node.nan_go_to == 2) {
						cur_tree_node = cur_tree_node.right_child;
					}
					else {
						// trainset has not missing value for this feature
						// so we should decide which branch the testset's missing value go to
						if (cur_tree_node.left_child.num_sample > cur_tree_node.right_child.num_sample) {
							cur_tree_node = cur_tree_node.left_child;
						}
						else {
							cur_tree_node = cur_tree_node.right_child;
						}
					}
				}
				else {
					if (feature[cur_tree_node.split_feature] <= cur_tree_node.split_threshold) {
						cur_tree_node = cur_tree_node.left_child;
					}
					else {
						cur_tree_node = cur_tree_node.right_child;
					}
				}
			}
			return cur_tree_node.leaf_score;
		}
	}
	
	private void clean_up() {
		this.alive_nodes = null;
	}
	
	private double sum(double[] vals) {
		double s = 0;
		for (double v : vals) s += v;
		return s;
	}
	
}
