package demon.main;

import java.util.Arrays;

public interface Loss {
	
	public double[] transform(double[] pred);
	public double[] grad(double[] pred, double[] label);
	public double[] hess(double[] pred, double[] label);	
}

class SquareLoss implements Loss{

	@Override
	public double[] grad(double[] pred, double[] label) {
		double[] ret = new double[pred.length];
		for (int i = 0; i < pred.length; ++i) {
			ret[i] = pred[i] - label[i];
		}
		return ret;
	}

	@Override
	public double[] hess(double[] pred, double[] label) {
		double[] ret = new double[pred.length];
		Arrays.fill(ret, 1.0);
		return ret;
	}

	@Override
	public double[] transform(double[] pred) {
		return pred;
	}
}

class LogisticLoss implements Loss{
	
	private double clip(double val) {
		if (val < 0.00001) {
			return 0.00001;
		}
		else if (val > 0.99999) {
			return 0.99999;
		}
		else {
			return val;
		}
	}
	
	public double[] transform(double[] pred) {
		double[] ret = new double[pred.length];
		for (int i = 0; i < ret.length; ++i) {
			ret[i] = clip(1.0 / (1.0 + Math.exp(-pred[i])));
		}
		return ret;
	}

	@Override
	public double[] grad(double[] pred, double[] label) {
		double[] pred_ = transform(pred);
		double[] ret = new double[pred_.length];
		for (int i = 0; i < ret.length; ++i) {
			ret[i] = pred_[i] - label[i];
		}
		return ret;
	}

	@Override
	public double[] hess(double[] pred, double[] label) {
		double[] pred_ = transform(pred);
		double[] ret = new double[pred_.length];
		for (int i = 0; i < ret.length; ++i) {
			ret[i] = pred_[i] * (1.0 - pred_[i]);
		}
		return ret;
	}
	
}
