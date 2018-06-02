package demon.main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Sampling {
	
	private Random rand;
	
	public Sampling() {}
	
	public Sampling(int seed) {
		rand = new Random(seed);
	}
	
	public <T> void shuffle(List<T> arras) {
		Collections.shuffle(arras, this.rand);
	}
	
	public Random getRandom() {
		return this.rand;
	}
}

class RowSampler extends Sampling{

	private List<Double> row_mask;
	
	public RowSampler(int seed) {
		super(seed);	
	}
	
	public RowSampler (int n, double sampling_rate) {
		super(0);
		row_mask = new ArrayList<>();
		for (int i = 0; i < n; ++i) {
			row_mask.add(this.getRandom().nextDouble() <= sampling_rate ? 1.0 : 0.0);
		}
	}
	
	public RowSampler(int n, double sampling_rate, int seed) {
		super(seed);
		row_mask = new ArrayList<>();
		for (int i = 0; i < n; ++i) {
			row_mask.add(this.getRandom().nextDouble() <= sampling_rate ? 1.0 : 0.0);
		}
	}
	
	public List<Double> getRowMask() {
		return this.row_mask;
	}
	
	public void shuffle() {
		super.shuffle(row_mask);
	}
}

class ColumnSampler extends Sampling{
	private List<Integer> cols;
	public List<Integer> col_selected;
	private int n_selected;
	
	public ColumnSampler(int n, double sampling_rate) {
		super(0);
		cols = new ArrayList<>();
		col_selected = new ArrayList<>();
		for (int i = 0; i < n; ++i) {
			cols.add(i);
		}
		n_selected = (int) (n * sampling_rate);
		col_selected = cols.subList(0, n_selected);
	}
	
	public ColumnSampler(int n, double sampling_rate, int seed) {
		super(seed);
		cols = new ArrayList<>();
		col_selected = new ArrayList<>();
		for (int i = 0; i < n; ++i) {
			cols.add(i);
		}
		n_selected = (int) (n * sampling_rate);
		col_selected = cols.subList(0, n_selected);
	}
	
	public void shuffle() {
		super.shuffle(cols);
		col_selected = cols.subList(0, n_selected);
	}
}