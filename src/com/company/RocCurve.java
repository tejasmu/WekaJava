package com.company;

import java.awt.*;
import java.io.*;
import java.util.*;
import javax.swing.*;

import weka.classifiers.functions.Logistic;
import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.*;
import weka.gui.visualize.*;

/**
 * Generates and displays a ROC curve from a dataset. Uses a default
 * NaiveBayes to generate the ROC data.
 *
 * @author FracPete
 */
public class RocCurve {

    /**
     * takes one argument: dataset in ARFF format (expects class to
     * be last attribute)
     */
    public static void main(String[] args) throws Exception {
        // load data
        ROCData();
    }

    public static void ROCData()
    {
        try {
            System.out.println("Roc Started");

            Instances testdata = null;
            Instances traindata = null;

            testdata = new Instances(
                    new BufferedReader(
                            new FileReader("F:\\Mani\\$ourceCode\\PyCtxbot_Modified\\ResultArff\\jonnykanyon\\469651447337193472.arff")));
            
            traindata = new Instances(
                    new BufferedReader(
                            new FileReader("F:\\Mani\\Experiments\\SmoteArff\\smote_usermodel1.arff")));

             
            testdata.setClassIndex(testdata.numAttributes() - 1);
            traindata.setClassIndex(traindata.numAttributes() - 1);

            // train classifier
            //ObjectInputStream ois = new ObjectInputStream(new FileInputStream("/home/mvijaya2/Experiments/Models/RegressionModels/logistic_usermodel4.model"));
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream("F:\\Tejas\\RLR\\Rm10.model.arff"));
            Logistic cl = (Logistic) ois.readObject();
            ois.close();


            //Classifier cl = new Logistic();
            //cl.buildClassifier(train);

            Evaluation eval = new Evaluation(traindata);
            eval.evaluateModel(cl,testdata);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println("generate curve");
            // generate curve

            ThresholdCurve tc = new ThresholdCurve();
            int classIndex = 1;
            Instances result = tc.getCurve(eval.predictions(), classIndex);
            System.out.println("plot curve");
            // plot curve
            ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
            vmc.setROCString("(Area under ROC = " +
                    Utils.doubleToString(tc.getROCArea(result), 4) + ")");
            vmc.setName(result.relationName());

            PlotData2D tempd = new PlotData2D(result);
            tempd.setPlotName(result.relationName());
            tempd.addInstanceNumberAttribute();
            // specify which points are connected
            boolean[] cp = new boolean[result.numInstances()];
            for (int n = 1; n < cp.length; n++)
                cp[n] = true;
            tempd.setConnectPoints(cp);
            // add plot
            vmc.addPlot(tempd);

            // display curve
            String plotName = vmc.getName();
            final javax.swing.JFrame jf =
                    new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
            jf.setSize(500,400);
            jf.getContentPane().setLayout(new BorderLayout());
            jf.getContentPane().add(vmc, BorderLayout.CENTER);
            jf.addWindowListener(new java.awt.event.WindowAdapter() {
                public void windowClosing(java.awt.event.WindowEvent e) {
                    jf.dispose();
                }
            });
            jf.setVisible(true);
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    /*
    public static void RocCurve()
    {
        try {
        System.out.println("Roc Started");
        Instances data = null;

            data = new Instances(
                    new BufferedReader(
                            new FileReader("F:\\Mani\\Experiments\\SmoteArff\\smote_usermodel1.arff")));

        data.setClassIndex(data.numAttributes() - 1);

        // train classifier
        Classifier cl = new Logistic();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(cl, data, 10, new Random(1));
        System.out.println("generate curve");
        // generate curve
        ThresholdCurve tc = new ThresholdCurve();
        int classIndex = 0;
        Instances result = tc.getCurve(eval.predictions(), classIndex);
        System.out.println("plot curve");
        // plot curve
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString("(Area under ROC = " +
                Utils.doubleToString(tc.getROCArea(result), 4) + ")");
        vmc.setName(result.relationName());

        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();
        // specify which points are connected
        boolean[] cp = new boolean[result.numInstances()];
        for (int n = 1; n < cp.length; n++)
            cp[n] = true;
        tempd.setConnectPoints(cp);
        // add plot
        vmc.addPlot(tempd);

        // display curve
        String plotName = vmc.getName();
        final javax.swing.JFrame jf =
                new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vmc, BorderLayout.CENTER);
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });
        jf.setVisible(true);
        }
        catch (IOException e) {
        e.printStackTrace();
         } catch (Exception e) {
            e.printStackTrace();
        }
    } 
    */
}