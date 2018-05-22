package main;

import main.transferfunctions.TransferFunction;

/**
 *
 * @author b2198
 */
public class MatrixOperations {
    private MatrixOperations(){}
    
    public static boolean isMatrix(double[][] matrix){
        int sizeCheck = -1;
        for(int i = 0; i < matrix.length; i++){
            if(i == 0){
                sizeCheck = matrix[i].length;
            } else {
                if(matrix[i].length != sizeCheck){
                    return false;
                }
            }
        }
        return true;
    }
    
    public static boolean compareDimensions(double[][] matrix1, double[][] matrix2){
        if(!isMatrix(matrix1) || !isMatrix(matrix2)){
            throw new RuntimeException("One of the arrays entered is not a matrix");
        }
        return matrix1.length == matrix2.length && matrix1[0].length == matrix2[0].length;
    }
    
    public static void applyTransferFunction(TransferFunction function, double[][] matrix, double[][] resultMatrix){
        if(!compareDimensions(matrix,resultMatrix)){
            throw new RuntimeException("The matrices don't have equal dimensions");
        }
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                resultMatrix[i][j] = function.f(matrix[i][j]);
            }
        }
    }
    
    public static double[][] applyTransferFunction(TransferFunction function, double [][] matrix){
        double[][] result = new double[matrix.length][matrix[0].length];
        applyTransferFunction(function, matrix, result);
        return result;
    }
    
    public static void applyTransferFunctionDerivative(TransferFunction function, double[][] matrix, double[][] resultMatrix){
        if(!compareDimensions(matrix,resultMatrix)){
            throw new RuntimeException("The matrices don't have equal dimensions");
        }
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                resultMatrix[i][j] = function.dfdx(matrix[i][j]);
            }
        }
    }
    
    public static double[][] applyTransferFunctionDerivative(TransferFunction function, double [][] matrix){
        double[][] result = new double[matrix.length][matrix[0].length];
        applyTransferFunctionDerivative(function, matrix, result);
        return result;
    }
    
    public static void elementByElementMult(double x, double[][] matrix, double[][] resultMatrix){
        if(!compareDimensions(matrix,resultMatrix)){
            throw new RuntimeException("The matrices don't have equal dimensions");
        }
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                resultMatrix[i][j] = matrix[i][j] * x;
            }
        }
    }
    
    public static double[][] elementByElementMult(double x, double[][] matrix){
        double[][] result = new double[matrix.length][matrix[0].length];
        elementByElementMult(x, matrix, result);
        return result;
    }
    
    public static void elementByElementMult(double[][] matrix1, double[][] matrix2, double[][] resultMatrix){
        if(!compareDimensions(matrix1,matrix2) || !compareDimensions(matrix1,resultMatrix)){
            throw new RuntimeException("The matrices don't have equal dimensions");
        }
        for(int i = 0; i < matrix1.length; i++){
            for(int j = 0; j < matrix1[i].length; j++){
                resultMatrix[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }
    }
    
    public static double[][] elementByElementMult(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix1[0].length];
        elementByElementMult(matrix1, matrix2, result);
        return result;
    }
    
    public static void matrixMult(double[][] matrix1, double[][] matrix2, double[][] resultMatrix){
        if(!isMatrix(matrix1) || !isMatrix(matrix2) || !isMatrix(resultMatrix)){
            throw new RuntimeException("One of the arrays entered is not a matrix");
        }
        if(matrix1[0].length != matrix2.length || matrix1.length != resultMatrix.length || matrix2[0].length != resultMatrix[0].length){
            throw new RuntimeException("multiplication not defined for the dimensions of the matrices or result matrix of incorrect dimensions:\n"
                    + "matrix1: " + matrix1.length + ", " + matrix1[0].length + "\n"
                    + "matrix2: " + matrix2.length + ", " + matrix2[0].length + "\n"
                    + "resultMatrix: " + resultMatrix.length + ", " + resultMatrix[0].length);
        }
        for(int i = 0; i < matrix1.length; i++){
            for(int j = 0; j < matrix2[0].length; j++){
                resultMatrix[i][j] = 0;
                for(int k = 0; k < matrix2.length; k++){
                    resultMatrix[i][j] += matrix1[i][k]*matrix2[k][j];
                }
            }
        }
    }
    
    public static double[][] matrixMult(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix2[0].length];
        matrixMult(matrix1, matrix2, result);
        return result;
    }
    
    public static void matrixSub(double[][] matrix1, double[][] matrix2, double[][] resultMatrix){
        if(!compareDimensions(matrix1, matrix2) || !compareDimensions(matrix1, resultMatrix)){
            throw new RuntimeException("The matrices don't have equal dimensions");
        }
        for(int i = 0; i < matrix1.length; i++){
            for(int j = 0; j < matrix1[i].length; j++){
                resultMatrix[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }
    }
    
    public static double[][] matrixSub(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix1[0].length];
        matrixSub(matrix1, matrix2, result);
        return result;
    }
    
    public static void matrixAdd(double[][] matrix1, double[][] matrix2, double[][] resultMatrix){
        if(!compareDimensions(matrix1, matrix2) || !compareDimensions(matrix1, resultMatrix)){
            throw new RuntimeException("The matrices don't have equal dimensions");
        }
        for(int i = 0; i < matrix1.length; i++){
            for(int j = 0; j < matrix1[i].length; j++){
                resultMatrix[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
    }
    
    public static double[][] matrixAdd(double[][] matrix1, double[][] matrix2){
        double[][] result = new double[matrix1.length][matrix1[0].length];
        matrixAdd(matrix1, matrix2, result);
        return result;
    }
    
    public static void transpose(double[][] matrix, double[][] resultMatrix){
        if(!isMatrix(matrix) || !isMatrix(resultMatrix)){
            throw new RuntimeException("One of the arrays entered is not a matrix");
        }
        if(matrix.length != resultMatrix[0].length || matrix[0].length != resultMatrix.length){
            throw new RuntimeException("Result matrix does not match the dimensions of matrix transpose");
        }
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                resultMatrix[j][i] = matrix[i][j];
            }
        }
    }
    
    public static double[][] transpose(double[][] matrix){
        double[][] result = new double[matrix[0].length][matrix.length];
        transpose(matrix, result);
        return result;
    }
    
    public static String toString(double[][] matrix){
        String string = "";
        if(!isMatrix(matrix)){
            throw new RuntimeException("The array entered is not a matrix");
        }
        for(int i = 0; i < matrix.length; i++){
            string += "| ";
            for(int j = 0; j < matrix[i].length; j++){
                string += "" + matrix[i][j] + "\t";
            }
            string += "|\n";
        }
        return string;
    }
    
    public static void copyTo(double[][] matrixToBeCopied, double[][] targetMatrix){
        if(!compareDimensions(matrixToBeCopied, targetMatrix)){
            throw new RuntimeException("The matrices don't have equal dimensions");
        }
        for(int i = 0; i < matrixToBeCopied.length; i++){
            for(int j = 0; j < matrixToBeCopied[i].length; j++){
                targetMatrix[i][j] = matrixToBeCopied[i][j];
            }
        }
    }
}
