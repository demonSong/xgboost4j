package demon.main;

import java.util.Arrays;
import java.util.List;

public class ClassList {

	private int data_size;
	private double[] label; //  根据 label 信息 和 预测信息，既能得到 grad 和 hess。未使用梯度下降，直接采取信息增益的手段
	
	public TreeNode[] corresponding_tree_node; // 每个用例 都能去到该去的结点
	private double[] pred;
	private double[] grad;  //每个样例 对应的 grad 和 hess
	private double[] hess;
	
	public ClassList(DMatrix data) {
		this.data_size = data.getDataSize();
		this.label = data.getLabel();
		this.pred = new double[data_size];
		this.grad = new double[data_size];
		this.hess = new double[data_size];
		this.corresponding_tree_node = new TreeNode[data_size];
	}
	
	public double[] getLabel() {
		return this.label;
	}
	
	public void initialize_pred(double first_round_pred) {
		Arrays.fill(pred, first_round_pred);
	}
	
	public void update_pred(double eta) {
		for (int i = 0; i < data_size; ++i) {
			pred[i] += eta * corresponding_tree_node[i].leaf_score;
		}
	}
	
	public void update_grad_hess(Loss loss, double scale_pos_weight) {
		grad = loss.grad(pred, label);
		hess = loss.hess(pred, label);
		if (scale_pos_weight != 1.0) {
			for (int i = 0; i < data_size; ++i) {
				if (label[i] == 1) {
					grad[i] *= scale_pos_weight;
					hess[i] *= scale_pos_weight;
				}
			}
		}
	}
	
	/**
	 * 每个feature，空值所在的下标
	 * @param missing_value_attribute_list
	 */
	public void update_grad_hess_missing_for_tree_node(int[][] missing_value_attribute_list) {
		for (int col = 0; col < missing_value_attribute_list.length; ++col) {
			for (int i : missing_value_attribute_list[col]) {
				TreeNode treeNode = corresponding_tree_node[i];
				if (!treeNode.isLeaf()) {
					treeNode.grad_missing[col] += grad[i]; // 
					treeNode.hess_missing[col] += hess[i];
				}
			}
		}
	}
	
	public void update_corresponding_tree_node(AttributeList attribute_list) {
		int lf_cnt = 0;
		int rt_cnt = 0;
		for (int i = 0; i < data_size; ++i) {
			TreeNode treeNode = corresponding_tree_node[i];
			if (!treeNode.isLeaf()) {
				int split_feature = treeNode.split_feature;
				double nan_go_to = treeNode.nan_go_to;
				double val = attribute_list.origin_feature[i].X[split_feature];
				double split_threshold = treeNode.split_threshold;
				
				if (DMatrix.isNAN(val)) {
					if (nan_go_to == 0) {
						corresponding_tree_node[i] = treeNode.nan_child;
					}
					else if (nan_go_to == 1) {
						corresponding_tree_node[i] = treeNode.left_child;
					}
					else {
						corresponding_tree_node[i] = treeNode.right_child;
					}
				}
				else if (val <= split_threshold) {
					corresponding_tree_node[i] = treeNode.left_child;
					lf_cnt ++;
				}
				else {
					rt_cnt ++;
					corresponding_tree_node[i] = treeNode.right_child;
				}
			}
		}
//		System.out.println("classList update_corresponding_tree_node :  " + lf_cnt + ", " + rt_cnt);
	}
	
	public void update_Grad_Hess_numsample_for_tree_node() {
		for (int i = 0; i < data_size; ++i) {
			TreeNode treeNode = corresponding_tree_node[i];
			if (!treeNode.isLeaf()) {
				treeNode.addGrad(grad[i]);
				treeNode.addHess(hess[i]);
				treeNode.addNumSample(1); //对结点进行instance计数
			}
		}
	}
	
	public void sampling(List<Double> row_mask) {
		for (int i = 0; i < data_size; ++i) {
			grad[i] *= row_mask.get(i);
			hess[i] *= row_mask.get(i);
		}
	}
	
	public double[] getPred() {
		return this.pred;
	}
	
	public double[] getGrad() {
		return this.grad;
	}
	
	public double[] getHess() {
		return this.hess;
	}
	
	public int getDataSize() {
		return this.data_size;
	}
	
}
