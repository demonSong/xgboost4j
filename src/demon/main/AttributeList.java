package demon.main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AttributeList {
	
	private int feature_dim;
	
	public double[][][] attribute_list; //记录了 值 和 值出现的instance 下标
	
	private int[][] missing_value_attribute_list;
	
	public int[][][] cutting_idx;
	public double[][] cutting_thresholds;
	
	public LabelPoint[] origin_feature;
	
	
	public AttributeList(DMatrix data) {
		this.missing_value_attribute_list = data.getMissingIndex();
		this.feature_dim = data.getFeatureDim();
		this.attribute_list = data.getFeatureValueIndex();
		this.origin_feature = data.getOriginFeature();
		
		sort_attribute_list();
		initialize_cutting_idx_thresholds();
		clean_up();
	}

	
	private void sort_attribute_list() {
		for (int i = 0; i < feature_dim; ++i) {
			Arrays.sort(attribute_list[i], (a, b) -> Double.compare(a[0], b[0]));
		}
	}
	
	private void initialize_cutting_idx_thresholds() {
		cutting_idx = new int[feature_dim][][];
		cutting_thresholds = new double[feature_dim][];
		
		for (int i = 0; i < feature_dim; ++i) {
			List<Integer> list = new ArrayList<>();
			int last_index = -1;
			for (int j = 0; j < attribute_list[i].length; ++j) {
				if (last_index == -1 || attribute_list[i][j][0] == attribute_list[i][last_index][0]) {
					last_index = j;
				}
				else {
					list.add(last_index);
					last_index = j;
				}
			}
			
			list.add(attribute_list[i].length - 1);
//			if (i == 1) {
//				int[] dd = {26988, 26991, 26993, 26994, 26995, 26999, 27000, 27001, 27009, 27010, 27025, 27026, 27029, 27030, 27057, 27064, 27066, 27067, 27143, 27144, 27145, 27146, 27147, 27165, 27166, 27173, 27175, 27176, 27177, 27178, 27362, 27364, 27365, 27368, 27373, 27375, 27396, 27397, 27398, 27399, 27401, 27467, 27468, 27472, 27494, 27495, 27502, 27503, 27508, 27509, 27511, 27512, 28214};
//				System.out.println("hello! initialize_cutting_idx_thresholds");
//				List<Double> vals = new ArrayList<>();
//				for (int j = 0; j < attribute_list[i].length; ++j) {
//					vals.add(attribute_list[i][j][0]);
//				}
//				for (int j : dd) {
//					System.out.println(attribute_list[i][j][0]);
//				}
//				System.out.println();
//			}
			
			cutting_thresholds[i] = new double[list.size()];
			for (int t = 0; t < cutting_thresholds[i].length; ++t) {
				cutting_thresholds[i][t] = attribute_list[i][list.get(t)][0];  //每个分割点的阈值
			}
			
			cutting_idx[i] = new int[list.size()][]; // 每个feature dim 中 有多少个分割点， 以及对应的idx
			
			int prv = 0;
			for (int k = 0; k < cutting_idx[i].length; ++k) {
				int end = list.get(k);
				int len = end - prv + 1;
				cutting_idx[i][k] = new int[len];
				for (int m = 0; m < len; ++m) {
					cutting_idx[i][k][m] = (int)attribute_list[i][prv + m][1];
				}
				prv = end + 1;
			}
			
//			for (int k = 0; k < cutting_idx[i].length; ++k) { //多少个分割点
//				int s_idx = list.get(k); // 当前
//				int e_idx = list.get(k + 1);
//				cutting_idx[i][k] = new int[e_idx - s_idx];
//				for (int m = 0; m < cutting_idx[i][k].length; ++m) {
//					cutting_idx[i][k][m] = (int)attribute_list[i][s_idx + m][1]; //下标
//				}
//			}
		}
	}
	
	public int[][] getMissingValueAttributeList(){
		return this.missing_value_attribute_list;
	}
	
	public int getFeatureDim() {
		return this.feature_dim;
	}
	
	private void clean_up() {
		attribute_list = null;
	}
	
	
}
