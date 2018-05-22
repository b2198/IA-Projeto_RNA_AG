package main.graphics;

import javafx.scene.canvas.Canvas;
import main.MLPArtificialNeuralNetwork.MLPListener;

/**
 *
 * @author b2198
 */
public abstract class MLPPlotter extends Canvas implements MLPListener {
    
    public MLPPlotter(){
        super(400,400);
    }
    
}
