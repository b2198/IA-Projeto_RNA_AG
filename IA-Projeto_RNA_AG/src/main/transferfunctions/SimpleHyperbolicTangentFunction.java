package main.transferfunctions;

/**
 *
 * @author b2198
 */
public class SimpleHyperbolicTangentFunction implements TransferFunction {

    @Override
    public double f(double x) {
        return Math.tanh(x);
    }

    @Override
    public double dfdx(double x) {
        double tanhx = Math.tanh(x);
        return 1-tanhx*tanhx;
    }
    
}
