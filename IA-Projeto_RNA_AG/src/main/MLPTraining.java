package main;

import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.embed.swing.SwingNode;
import javafx.scene.Scene;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import main.graphics.DatasetWriter;
import main.graphics.MLPPlotter;
import main.graphics.EpochErrorMLPPlotter;
import main.transferfunctions.SimpleSigmoidFunction;
import main.transferfunctions.TransferFunction;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.DefaultXYDataset;

/**
 *
 * @author b2198
 */
public class MLPTraining extends Application {
    
    public static void main(String[] args){
        launch(args);
    }
    
    
    @Override
    public void start(Stage primaryStage) throws Exception {
        
        DefaultXYDataset dataset = new DefaultXYDataset();
        DatasetWriter writer = new DatasetWriter(dataset);
        ValueAxis domainAxis = new NumberAxis();
        ValueAxis rangeAxis = new NumberAxis();
        XYItemRenderer renderer = new XYLineAndShapeRenderer(true, false);
        Plot plot = new XYPlot(dataset, domainAxis, rangeAxis, renderer);
        java.awt.Color graphBG = new java.awt.Color(220, 255, 220);
        plot.setBackgroundPaint(graphBG);
        JFreeChart chart = new JFreeChart("test chart", plot);
        ChartPanel cPanel = new ChartPanel(chart, 600, 600, 400, 400, 500, 500, true, true, true, true, true, true);
        SwingNode sNode = new SwingNode();
        sNode.setContent(cPanel);
        
        VBox vbox = new VBox(sNode);
        StackPane root = new StackPane(vbox);
        Scene scene = new Scene(root,400,400);
        primaryStage.setScene(scene);
        primaryStage.show();
        
        double[][] input = new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int[] hiddenAmount = new int[]{3,3};
        double[][] expectedOutput = new double[][]{
            {0,0},
            {1,1},
            {1,1},
            {0,1}
        };
        TransferFunction function;
        function = new SimpleSigmoidFunction(1.0,-0.00);
        //function = new SimpleHyperbolicTangentFunction();
        double learningRate = 0.4;
        double momentum = 0.8;
        MLPArtificialNeuralNetwork xor = new MLPArtificialNeuralNetwork(input, hiddenAmount, expectedOutput, function, learningRate, momentum);
        xor.initializeRandomWeights();
        xor.getListeners().add(writer);
        
        double[][] testInput = new double[][]{
            {0,0}
        };
        
        int maxEpochNumber = 10000000;
        double minAverageError = 0.001;
        startTraining(xor, testInput, maxEpochNumber, minAverageError);
        
    }
    
    private void startTraining(MLPArtificialNeuralNetwork mlp, double[][] testInput, int maxEpochNumber, double minAverageError){
        Task trainTask = new Task(){
            @Override
            protected String call() throws Exception {
                mlp.train(maxEpochNumber,minAverageError);
                return "trained";
            }

            @Override
            protected void succeeded() {
                mlp.test(testInput);
            }

            @Override
            protected void failed() {
                System.out.println("something has gone wrong");
                startTraining(mlp, testInput, maxEpochNumber, minAverageError);
            }
            
            
        };
        Thread thread = new Thread(trainTask);
        thread.setDaemon(true);
        thread.start();
    }
    
}
