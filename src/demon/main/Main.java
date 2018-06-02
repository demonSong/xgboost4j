package demon.main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class Main {
	
	public static Log log = LogFactory.getLog(Main.class);
	
	static class DataFrame{
		double[][] features;
		double[] label;
		public DataFrame(double[][] features, double[] label){
			this.features = features;
			this.label = label;
		}
	}
	
	public static DataFrame readCsvFeature(String filename, int label_index, int[] ignore_index){
		try {
			BufferedReader in = new BufferedReader(new FileReader(filename));
			String line = "";
			int cnt = 0;
			List<String[]> datas = new ArrayList<>();
			while ((line = in.readLine()) != null) {
				cnt ++;
				if (cnt == 1) {
					continue;
				}
				String[] data = line.split(",");
				datas.add(data);
			}
			in.close();
			
			Set<Integer> ignore_idx = new HashSet<>();
			for (int i = 0; i < ignore_index.length; ++i) ignore_idx.add(ignore_index[i]);
			ignore_idx.add(label_index);
			
			double[][] ret = new double[datas.size()][];
			double[] label = new double[datas.size()];
			for (int i = 0; i < datas.size(); ++i) {
				String[] data = datas.get(i);
				int featureDim = data.length - 1 - ignore_index.length;
				double[] fe = new double[featureDim];
				for (int j = 0, k = 0; j < data.length; ++j) {
					if (ignore_idx.contains(j)) continue;
					fe[k++] = Double.parseDouble(data[j]);
				}
				label[i] = Double.parseDouble(data[label_index]);
				ret[i] = fe;
			}
			
			return new DataFrame(ret, label);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	static class Split{
		DataFrame train;
		DataFrame valid;
		
		public Split(double valid_ratio, DataFrame dataset){
			int size = dataset.label.length;
			int test_size = (int) (valid_ratio * size);
			double[][] train_X = new double[size - test_size][];
			double[][] valid_X = new double[test_size][];
			
			double[] train_y = new double[size - test_size];
			double[] valid_y = new double[test_size];
			
			for (int i = 0, tc = 0, vc = 0; i < size; ++i) {
				if (i < test_size) {
					valid_X[vc] = dataset.features[i];
					valid_y[vc] = dataset.label[i];
					vc ++;
				}
				else {
					train_X[tc] = dataset.features[i];
					train_y[tc] = dataset.label[i];
					tc ++;
				}
			}
			
			this.train = new DataFrame(train_X, train_y);
			this.valid = new DataFrame(valid_X, valid_y);
		}
	}
	
	public static void persist(DataFrame data, String filename) {
		StringBuilder sb = new StringBuilder();
		double[][] features = data.features;
		double[] label = data.label;
		int data_size = label.length;
		int feature_dim = features[0].length;
		List<String> head = new ArrayList<>();
		for (int j = 0; j < feature_dim; ++j) head.add("col" + j);
		head.add("label");
		sb.append(String.join(",", head.toArray(new String[head.size()])) + "\n");
		for (int i = 0; i < data_size; ++i) {
			List<String> line = new ArrayList<>();
			for (int j = 0; j < feature_dim; ++j) {
				line.add(String.valueOf(features[i][j]));
			}
			line.add(String.valueOf(label[i]));
			String[] Xy = line.toArray(new String[line.size()]);
			sb.append(String.join(",", Xy) + "\n");
		}
		try {
			Files.write(Paths.get(filename), sb.toString().getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) throws IOException {
		DataFrame dd = readCsvFeature("./data/TR_01031509.csv", 3, new int[] {0});
		Split sp = new Split(0.3, dd);
		
//		persist(sp.train, "./data/dc_train.csv");
//		persist(sp.valid, "./data/dc_valid.csv");
		
		XGBoost model = new XGBoost();
		DMatrix train = new DMatrix(sp.train.features, sp.train.label);
		DMatrix valid = new DMatrix(sp.valid.features, sp.valid.label);
		
		Map<String, DMatrix> dataset = new HashMap<>();
		dataset.put("train", train);
		dataset.put("valid", valid);
		
		model.fit(0.1, 2000, 8, 50, 1, 0.8, 0.8, 1, 0, 20, 0, true, 4, "auc", "logloss", dataset);
		
		ModelSerializer.save_model(model, "./data/tree.model");
	}
}
