package demon.main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TreeNode {
	
	public int index;
	public int depth;
	public int feature_dim;
	private boolean is_leaf;
	public int num_sample;
	
	private double[] G_left;
	private double[] H_left;
	
	private double[] best_thresholds;
	public double[] best_gains;  // col 的 best gain
	private double[] best_nan_go_to;
	
	public double nan_go_to;
	
	public double[] grad_missing;
	public double[] hess_missing;
	
	public double grad;
	public double hess;
	
	// leaf node
	public double leaf_score;
	
	// internal node
	public int split_feature;
	public double split_threshold;
	public List<Double> split_left_child_catvalue;
	public TreeNode nan_child;
	public TreeNode left_child;
	public TreeNode right_child;
	
	
	public TreeNode(int index, int depth, int feature_dim, boolean is_leaf) {
		this.index = index;
		this.depth = depth;
		this.feature_dim = feature_dim;
		this.is_leaf = is_leaf;
		this.G_left = new double[feature_dim];
		this.H_left = new double[feature_dim];
		this.best_thresholds = new double[feature_dim];
		this.best_gains = new double[feature_dim];
		this.best_nan_go_to = new double[feature_dim];
		this.grad_missing = new double[feature_dim];
		this.hess_missing = new double[feature_dim];
		Arrays.fill(this.best_gains, -Double.MAX_VALUE);
	}
	
	public void update_best_split(int col, double threshold, double gain, double nan_go_to) {
		if (gain > best_gains[col]) {
			best_gains[col] = gain;
			best_thresholds[col] = threshold;
			best_nan_go_to[col] = nan_go_to;
		}
	}
	
	public List<Double> get_best_feature_threshold_gain(){
		int best_feature = 0;
		double max_gain = -Double.MAX_VALUE;
		for (int i = 0; i < feature_dim; ++i) {
			if (best_gains[i] > max_gain) {
				max_gain = best_gains[i];
				best_feature = i;
			}
		}
		
		List<Double> ret = new ArrayList<>();
		ret.add((double) best_feature);
		ret.add(max_gain);
		ret.add(best_nan_go_to[best_feature]);
		ret.add(best_thresholds[best_feature]);
		return ret;
	}
	
	public void setLeafNode(double leaf_score, boolean isLeaf) {
		this.is_leaf = isLeaf;
		this.leaf_score = leaf_score;
		clean_up();
	}
	
	public void setInternalNode(double feature, double threshold, double nan_go_to,
			TreeNode nan_child, TreeNode left_child, TreeNode right_child, boolean is_leaf) {
		this.split_feature = (int) feature;
		this.split_threshold = threshold;
		this.nan_go_to = nan_go_to;
		this.nan_child = nan_child;
		this.left_child = left_child;
		this.right_child = right_child;
		this.is_leaf = is_leaf;
		clean_up();
	}
	
	public void addNumSample(double value) {
		num_sample += value;
	}
	
	public void addGrad(double value) {
		this.grad += value;
	}
	
	public void addHess(double value) {
		this.hess += value;
	}
	
	public void setGrad(double value) {
		this.grad = value;
	}
	
	public void setHess(double hess) {
		this.hess = hess;
	}
	
	public double[] getGLeft() {
		return this.G_left;
	}
	
	public double[] getHLeft() {
		return this.H_left;
	}
	
	public boolean isLeaf() {
		return this.is_leaf;
	}
	
	private void clean_up() {
		best_thresholds = null;
		best_gains = null;
		best_nan_go_to = null;
		G_left = null;
		H_left = null;
	}
}
