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
import main.graphics.MLPConectionRenderer;
import main.transferfunctions.SimpleHyperbolicTangentFunction;
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
        
        MLPConectionRenderer conections = new MLPConectionRenderer();
        
        DefaultXYDataset dataset = new DefaultXYDataset();
        DatasetWriter writer = new DatasetWriter(dataset);
        ValueAxis domainAxis = new NumberAxis();
        ValueAxis rangeAxis = new NumberAxis();
        XYItemRenderer renderer = new XYLineAndShapeRenderer(true, false);
        Plot plot = new XYPlot(dataset, domainAxis, rangeAxis, renderer);
        java.awt.Color graphBG = new java.awt.Color(220, 255, 220);
        plot.setBackgroundPaint(graphBG);
        JFreeChart chart = new JFreeChart("gráfico erro x época", plot);
        ChartPanel cPanel = new ChartPanel(chart, 600, 600, 400, 400, 500, 500, true, true, true, true, true, true);
        SwingNode sNode = new SwingNode();
        sNode.setContent(cPanel);
        
        VBox vbox1 = new VBox(conections);
        VBox vbox2 = new VBox(sNode);
        vbox2.setPrefSize(500, 500);
        HBox hbox = new HBox(vbox1, vbox2);
        StackPane root = new StackPane(hbox);
        Scene scene = new Scene(root);
        primaryStage.setScene(scene);
        primaryStage.show();
        
        
        //MLP specific
        
        
        double[][] input = new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        int[] hiddenAmount = new int[]{4,1};
        double[][] expectedOutput = new double[][]{
            {0},
            {1},
            {1},
            {0}
        };
        TransferFunction function;
        //function = new SimpleSigmoidFunction(1.1,-0.05);
        function = new SimpleHyperbolicTangentFunction();
        double learningRate = 0.5;
        double momentum = 0.83;
        MLPArtificialNeuralNetwork xor = new MLPArtificialNeuralNetwork(input, hiddenAmount, expectedOutput, function, learningRate, momentum);
        xor.initializeRandomWeights();
        xor.getListeners().add(conections);
        xor.getListeners().add(writer);
        
        double[][] testInput = new double[][]{
            {0,0}
        };
        
        int maxEpochNumber = 100000;
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
