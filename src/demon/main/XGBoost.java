package demon.main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;


public class XGBoost {
	public static Log logger = LogFactory.getLog(XGBoost.class);
	
	private List<Tree> trees = new ArrayList<>();
	private double eta;
	private double rowsample;
	private double colsample;
	private double lambda;
	private double gamma;
	private double min_child_weight;
	private double scale_pos_weight;
	
	private int max_depth;
	private int num_boost_round;
	private int min_sample_split;
	private int num_thread;
	
	private double first_round_pred;
	
	private String eval_metric;
	private Loss loss;
	
	public XGBoost() {}
	public XGBoost(List<Tree> trees, Loss loss, double first_round_pred, double eta) {
		this.trees = trees;
		this.loss = loss;
		this.first_round_pred = first_round_pred;
		this.eta = eta;
	}
	
	public void fit(double eta, 
					   int num_boost_round,
					   int max_depth,
					   int early_stopping_rounds,
					   int min_sample_split,
					   double scale_pos_weight,
					   double rowsample,
					   double colsample,
					   double min_child_weight,
					   double lambda,
					   double gamma,
					   boolean maximize,
					   int num_thread,
					   String eval_metric,
					   String loss,
					   Map<String, DMatrix> dataset) {
		
		this.eta = eta;
		this.num_boost_round = num_boost_round;
		this.max_depth = max_depth;
		this.rowsample = rowsample;
		this.colsample = colsample;
		this.lambda = lambda;
		this.gamma = gamma;
		this.min_child_weight = min_child_weight;
		this.min_sample_split = min_sample_split;
		this.num_thread = num_thread;
		this.eval_metric = eval_metric;
		this.scale_pos_weight = scale_pos_weight;
		
		DMatrix trainset = dataset.get("train");
		DMatrix validset = dataset.getOrDefault("valid", null);
		
		AttributeList attribute_list = new AttributeList(trainset);
		ClassList class_list = new ClassList(trainset);
		
		RowSampler row_sampler = new RowSampler(trainset.getDataSize(), this.rowsample);
		ColumnSampler col_sampler = new ColumnSampler(trainset.getFeatureDim(), this.colsample);
		
		trainset = null;
		
		if (loss.equals("logloss")) {
			this.loss = new LogisticLoss();
			this.first_round_pred = 0.0;
		}
		else if (loss.equals("squareloss")){
			this.loss = new SquareLoss();
			this.first_round_pred = average(class_list.getLabel());
		}
		else {
			throw new UnsupportedOperationException();
		}
		
		class_list.initialize_pred(this.first_round_pred);
		class_list.update_grad_hess(this.loss, this.scale_pos_weight);
		
		boolean do_validation;
		double[] val_pred;
		if (validset == null) {
			do_validation = false;
			val_pred = null;
		}
		else {
			do_validation = true;
			val_pred = new double[validset.getDataSize()];
			Arrays.fill(val_pred, this.first_round_pred);
		}
		
		double best_val_metric;
		int best_round = 0;
		int become_worse_round = 0;
		if (maximize) {
			best_val_metric = -Double.MAX_VALUE;
		}
		else {
			best_val_metric = Double.MAX_VALUE;
		}
		
		logger.info("TinyXGBoost start training");
		for (int i = 0; i < this.num_boost_round; ++i) {
			Tree tree = new Tree(min_sample_split, min_child_weight, max_depth,
					colsample, rowsample, lambda, gamma, num_thread);
			tree.fit(attribute_list, class_list, row_sampler, col_sampler);
			
			// eta 作用在了叶子结点上，作为加权系数
			class_list.update_pred(this.eta);
			class_list.update_grad_hess(this.loss, this.scale_pos_weight);
			
			// save this tree
			this.trees.add(tree);
			logger.info(String.format("current tree has %d nodes,including %d nan tree nodes",tree.nodes_cnt,tree.nan_nodes_cnt));
			
			if (eval_metric.equals("")) {
				logger.info(String.format("TGBoost round %d",i));
			}
			else {
				double train_metric = calculate_metric(eval_metric, this.loss.transform(class_list.getPred()), class_list.getLabel());
				if (!do_validation) {
					logger.info(String.format("TGBoost round %d, train-%s:%.6f", i, eval_metric, train_metric));
				}
				else {
					double[] cur_tree_pred = tree.predict(validset);
					for (int n = 0; n < val_pred.length; ++n) {
						val_pred[n] += this.eta * cur_tree_pred[n];
					}
					double val_metric = calculate_metric(eval_metric, this.loss.transform(val_pred), validset.getLabel());
					logger.info(String.format("TGBoost round %d,train-%s:%.6f,val-%s:%.6f",i,eval_metric,train_metric,eval_metric,val_metric));
					if (maximize) {
						if (val_metric > best_val_metric) {
							best_val_metric = val_metric;
							best_round = i;
							become_worse_round = 0;
						}
						else {
							become_worse_round +=1;
						}
						if (become_worse_round > early_stopping_rounds) {
							logger.info(String.format("TGBoost training stop,best round is %d,best val-%s is %.6f",i,eval_metric,best_val_metric));
							break;
						}
					}
					else {
						if(val_metric < best_val_metric){
                            best_val_metric = val_metric;
                            best_round = i;
                            become_worse_round = 0;
                        }else {
                            become_worse_round += 1;
                        }
                        if(become_worse_round>early_stopping_rounds){
                            logger.info(String.format("TGBoost training stop,best round is %d,best val-%s is %.6f",i,eval_metric,best_val_metric));
                            break;
                        }
					}
				}
			}
		}
	}
	
	
	public List<Tree> getTrees(){
		return this.trees;
	}
	
	public double getEta() {
		return this.eta;
	}
	
	public Loss getLoss() {
		return this.loss;
	}
	
	public double geFirstRoundPred() {
		return this.first_round_pred;
	}
	
	private double calculate_metric(String eval_metric, double[] pred, double[] label) {
		if (eval_metric.equals("acc")){
			return Metric.accuracy(pred, label);
		}
		else if (eval_metric.equals("error")) {
			return Metric.error(pred, label);
		}
		else if (eval_metric.equals("mse")){
			return Metric.mean_square_error(pred, label);
		}
		else if (eval_metric.equals("mae")) {
			return Metric.mean_absolute_error(pred, label);
		}
		else if (eval_metric.equals("auc")) {
			return Metric.auc(pred, label);
		}
		else {
			throw new UnsupportedOperationException();
		}
	}
	
	private double average(double[] vals) {
		double sum = 0.0;
		for (double v : vals)  sum += v;
		return sum / vals.length;
	}

}
