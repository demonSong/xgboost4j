package demon.main;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class ModelSerializer {
	
	private static String serializeLeafNode(TreeNode node) {
		StringBuilder sb = new StringBuilder();
		sb.append(node.index);
		sb.append(":leaf=");
		sb.append(String.format("%.6f", node.leaf_score));
		return sb.toString();
	}
	
	private static String serializeInternalNode(TreeNode node) {
		StringBuilder sb = new StringBuilder();
		sb.append(node.index);
		sb.append(":[");
		sb.append(node.split_feature + ",");
		
		if (node.split_left_child_catvalue == null){
			sb.append("num,");
			sb.append(String.format("%.6f", node.split_threshold));
			sb.append("],");
		}
		
		if(node.nan_go_to==0){
            sb.append("missing_go_to=0");
        }else if(node.nan_go_to==1){
            sb.append("missing_go_to=1");
        }else if(node.nan_go_to==2){
            sb.append("missing_go_to=2");
        }else{
            if(node.left_child.num_sample>node.right_child.num_sample){
                sb.append("missing_go_to=1");
            }else {
                sb.append("missing_go_to=2");
            }
        }
        return sb.toString();
	}
	
	public static void save_model(XGBoost xgb, String path) { 
		double first_round_predict = xgb.geFirstRoundPred();
		double eta = xgb.getEta();
		Loss loss = xgb.getLoss();
		List<Tree> trees = xgb.getTrees();
		
		StringBuilder sb = new StringBuilder();
		sb.append("first_round_predict=" + first_round_predict + "\n");
		sb.append("eta=" + eta + "\n");
		if (loss instanceof LogisticLoss) {
			sb.append("logloss" + "\n");
		}
		else {
			sb.append("squareloss" + "\n");
		}
		
		for (int i = 1; i < trees.size(); ++i) {
			sb.append("tree[" + i + "]:\n");
			Tree tree = trees.get(i - 1);
			TreeNode root = tree.getRoot();
			Queue<TreeNode> queue = new LinkedList<>();
			queue.offer(root);
			while (!queue.isEmpty()) {
				int cur_level_num = queue.size();
				for (int j = 0; j < cur_level_num; ++j) {
					TreeNode node = queue.poll();
					if (node.isLeaf()) {
						sb.append(serializeLeafNode(node) + "\n");
					}
					else {
						sb.append(serializeInternalNode(node) + "\n");
						queue.offer(node.left_child);
						if (node.nan_child != null) {
							queue.offer(node.nan_child);
						}
						queue.offer(node.right_child);
					}
				}
			}
		}
		sb.append("tree[end]");
		
		try {
			Files.write(Paths.get(path), sb.toString().getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
