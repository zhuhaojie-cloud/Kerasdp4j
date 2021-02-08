package com.demo.keras;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
public class App
{
    public static void main( String[] args )
    {
        try {
            String path = new ClassPathResource("classv.h5").getFile().getPath();
            MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(path);
            INDArray input = Nd4j.create(new float[]{0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1}, new int[]{1, 10});
            System.out.println("Input: " + input);
            double prediction = model.output(input).getDouble(0);
            System.out.println("Output: " + prediction);
        } catch (IOException | UnsupportedKerasConfigurationException | InvalidKerasConfigurationException e) {
            e.printStackTrace();
        }

    }
}
