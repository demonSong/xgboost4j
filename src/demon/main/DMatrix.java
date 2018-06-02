package demon.main;

import java.util.Arrays;

public class DMatrix {
	
	private double[][][] feature_value_idx;
	private LabelPoint[] origin_features;
	private double[] label;
	
	private int[] missing_cnt;
	private int[][] missing_idx;
	
	
	private int data_size;
	private int feature_dim;
	
	public static double NAN = -Double.MAX_VALUE;
	
	public DMatrix(double[][] X, double[] y) {
		this.label = y;
		this.data_size = X.length;
		this.feature_dim = X[0].length;
		this.missing_cnt = new int[feature_dim];
		
		for (int i = 0; i < feature_dim; ++i) {
			for (int j = 0; j < data_size; ++j) {
				double v = X[j][i];
				if (isNAN(v)) missing_cnt[i]++;
			}
		}
		
		missing_idx = new int[feature_dim][];
		feature_value_idx = new double[feature_dim][][];
		
		for (int i = 0; i < feature_dim; ++i) {
			int cnt = missing_cnt[i];
			missing_idx[i] = new int[cnt];
			feature_value_idx[i] = new double[data_size - cnt][2];
		}
		
		origin_features = new LabelPoint[data_size];
		
		int[] cur_idx = new int[feature_dim];
		int[] cur_missing_idx = new int[feature_dim];
		Arrays.fill(cur_idx, 0);
		Arrays.fill(cur_missing_idx, 0);
		
		for (int i = 0; i < data_size; ++i) {
			origin_features[i] = new LabelPoint();
			origin_features[i].y = y[i];
			
			double[] X_ = new double[feature_dim];
			for (int j = 0; j < feature_dim; ++j) {
				double v = X[i][j];
				if (isNAN(v)) {
					missing_idx[j][cur_missing_idx[j]] = i;
					cur_missing_idx[j] += 1;
					X_[j] = NAN;
				}
				else {
					feature_value_idx[j][cur_idx[j]][0] = X[i][j];
					feature_value_idx[j][cur_idx[j]][1] = i;
					cur_idx[j] += 1;
					X_[j] = X[i][j];
				}
			}
			origin_features[i].X = X_;
		}
		
	}
	
	public static boolean isNAN(double x) {
		return Double.compare(NAN, x) == 0;
	}
	
	public int[][] getMissingIndex(){
		return this.missing_idx;
	}
	
	public int getFeatureDim() {
		return this.feature_dim;
	}
	
	public double[][][] getFeatureValueIndex(){
		return this.feature_value_idx;
	}
	
	public LabelPoint[] getOriginFeature() {
		return this.origin_features;
	}
	
	public int getDataSize() {
		return this.data_size;
	}
	
	public double[] getLabel() {
		return this.label;
	}
}


class LabelPoint{
	public double[] X;
	public double y;
	
	LabelPoint(){}
	
	LabelPoint(double[] X, double y){
		this.X = X;
		this.y = y;
	}
	
}