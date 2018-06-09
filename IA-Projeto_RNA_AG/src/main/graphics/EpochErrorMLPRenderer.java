package main.graphics;

import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import main.MLPArtificialNeuralNetwork;

/**
 *
 * @author b2198
 */
public class EpochErrorMLPRenderer extends MLPRenderer {
    
    private double xScale, yScale;
    private double lastTime, lastError;
    private long lastRealTime;
    private long currentVisualCycle;

    public EpochErrorMLPRenderer(double xScale, double yScale) {
        this.xScale = xScale;
        this.yScale = yScale;
    }
    
    @Override
    public void onTrainingStarted(MLPArtificialNeuralNetwork.MLPInfo info) {
        GraphicsContext gc = getGraphicsContext2D();
        gc.setFill(new Color(0.92,0.97,0.92,1));
        gc.fillRect(0, 0, getWidth(), getHeight());
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(10);
        gc.strokeRect(0, 0, getWidth(), getHeight());
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(1);
        gc.strokeLine(0, 0+getHeight()/2, 0+Integer.MAX_VALUE, 0+getHeight()/2);
        gc.strokeLine(0+getWidth()/2, 0, 0+getWidth()/2, 0+getHeight());
        gc.translate(getWidth()/2, getHeight()/2);
        gc.scale(xScale, yScale);
        gc.translate(-getWidth()/2, -getHeight()/2);
        lastRealTime = System.nanoTime();
        currentVisualCycle = 0;
        gc.save();
    }

    @Override
    public void onCycleStart(MLPArtificialNeuralNetwork.MLPInfo info, int cycle) {
        
    }

    @Override
    public void onCycleEnd(MLPArtificialNeuralNetwork.MLPInfo info, int cycle, double averageError) {
        if(System.nanoTime()-lastRealTime > 1e9/1000.0){
            lastRealTime = System.nanoTime();
            GraphicsContext gc = getGraphicsContext2D();
            if(currentVisualCycle == 0){
                gc.translate(getWidth()/2, getHeight()/2);
                gc.scale(1/xScale, 1/yScale);
                gc.translate(-getWidth()/2, -getHeight()/2);
                gc.strokeText(String.format("%.5f", averageError), 0+getWidth()/2-50, getHeight()/2-averageError*yScale);
                gc.restore();
            }
            gc.setFill(Color.BLACK);
            gc.fillOval(currentVisualCycle+getWidth()/2-1, getHeight()/2-averageError-1, 2/xScale, 2/yScale);
            currentVisualCycle++;
        }
    }

    @Override
    public void onFinished(MLPArtificialNeuralNetwork.MLPInfo info, int finalEpoch, boolean success, double finalError) {
        GraphicsContext gc = getGraphicsContext2D();
        gc.translate(getWidth()/2, getHeight()/2);
        gc.scale(1/xScale, 1/yScale);
        gc.translate(-getWidth()/2, -getHeight()/2);
        gc.strokeText(String.format("%.5f", finalError), 0+getWidth()/2-50, getHeight()/2-finalError*yScale);
    }
    
}
