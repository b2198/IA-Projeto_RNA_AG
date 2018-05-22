package main.graphics;

import java.util.LinkedList;
import main.MLPArtificialNeuralNetwork;
import main.MLPArtificialNeuralNetwork.MLPListener;
import org.jfree.data.xy.DefaultXYDataset;

/**
 *
 * @author b2198
 */
public class DatasetWriter implements MLPListener {
    
    private DefaultXYDataset dataset;
    private LinkedList<Double> ys;
    
    public DatasetWriter(DefaultXYDataset dataset){
        this.dataset = dataset;
        ys = new LinkedList<>();
    }

    @Override
    public void onTrainingStarted(MLPArtificialNeuralNetwork.MLPInfo info) {
    }

    @Override
    public void onCycleStart(MLPArtificialNeuralNetwork.MLPInfo info, int cycle) {
    }

    @Override
    public void onCycleEnd(MLPArtificialNeuralNetwork.MLPInfo info, int cycle, double averageError) {
        ys.add(averageError);
    }

    @Override
    public void onFinished(MLPArtificialNeuralNetwork.MLPInfo info, int finalEpoch, boolean success, double finalError) {
        double[] xs = new double[this.ys.size()];
        double[] ys = new double[this.ys.size()];
        for(int i = 0; i < xs.length; i++){
            xs[i] = i;
            ys[i] = this.ys.get(i);
        }
        double[][] data = new double[][]{
            xs, ys
        };
        dataset.addSeries("error", data);
    }
    
}
