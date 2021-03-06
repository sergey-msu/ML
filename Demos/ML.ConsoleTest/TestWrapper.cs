﻿using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core;
using ML.Core.Kernels;
using ML.Core.Logical;
using ML.Core.Metric;
using ML.Registry;
using ML.Contracts;
using ML.LogicalMethods.Algorithms;
using ML.MetricMethods.Algorithms;
using ML.NeuralMethods.Algorithms;
using ML.DeepMethods.Models;
using ML.DeepMethods.Registry;
using ML.Utils;
using ML.BayesianMethods.Algorithms;
using ML.Core.Distributions;

namespace ML.ConsoleTest
{
  public class TestWrapper
  {
    #region Inner

    public class Runner : IDisposable
    {
      private readonly System.Diagnostics.Stopwatch m_Stopwatch;

      public Runner()
      {
        m_Stopwatch = new System.Diagnostics.Stopwatch();
        Console.WriteLine("RUNNER STARTED");
        m_Stopwatch.Start();
      }

      public void Dispose()
      {
        m_Stopwatch.Stop();
        Console.WriteLine("RUNNER STOPPED at [{0}] s", m_Stopwatch.Elapsed.TotalSeconds);
      }
    }

    #endregion

    public TestWrapper(DataWrapper data)
    {
      if (data == null)
        throw new ArgumentException("TestWrapper.ctor(data=null)");

      Data = data;
      Visualizer = new Visualizer(data);
    }

    public readonly DataWrapper Data;
    public readonly Visualizer  Visualizer;


    public void Run()
    {
      using (var runner = new Runner())
      {
        var start = DateTime.Now;

        //testSerDeser();

        //doBayesianParzenAlgorithmTest();
        //doNaiveBayesianKernelAlgorithmTest();
        //doBayesianKernelAlgorithmTest();
        //doQuadraticDiscriminantAlgorithmTest();
        doNaiveBayesianGeneralAlgorithmTest();

        //doNearestNeighbourAlgorithmTest();
        //doNearestKNeighboursAlgorithmTest();
        //doParzenFixedAlgorithmTest();
        //doPotentialFixedAlgorithmTest();

        //doDecisionTreeAlgorithmTest();

        //doMultilayerNNAlgorithmTest();
        //doCNNAlgorithmTest();

        var stop = DateTime.Now;
        Console.WriteLine("elapsed: "+(stop-start).TotalMilliseconds);
      }
    }

    #region .pvt

    private void testSerDeser()
    {
      //var activation = Activation.ReLU;
      //var net = new ConvNet(1, 28) { IsTraining=true };
      //
      //net.AddLayer(new ConvLayer(outputDepth: 8, windowSize: 5));
      //net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      //net.AddLayer(new ConvLayer(outputDepth: 18, windowSize: 5));
      //net.AddLayer(new MaxPoolingLayer(windowSize: 2, stride: 2, activation: activation));
      //net.AddLayer(new FlattenLayer(outputDim: 10, activation: activation));
      //
      //net._Build();
      //
      //net.RandomizeParameters(seed: 0);

      var path = @"C:\Users\User\Desktop\net.mld";
      //using (var file = System.IO.File.Open(path, System.IO.FileMode.Create, System.IO.FileAccess.Write))
      //  net.Serialize(file);
      using (var file = System.IO.File.Open(path, System.IO.FileMode.Open, System.IO.FileAccess.Read))
      {
        var res = ConvNet.Deserialize(file);
      }
    }


    private void doNearestNeighbourAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var alg = new NearestNeighbourAlgorithm(metric);
      alg.Train(Data.TrainingSample);

      Console.WriteLine("Margin:");
      calculateMargin(alg);

      Console.WriteLine("Errors:");
      var errors = alg.GetErrors(Data.Data, 0, true);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);

      Visualizer.Run(alg);
    }

    private void doNearestKNeighboursAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var alg = new NearestKNeighboursAlgorithm(metric, 1);
      alg.Train(Data.TrainingSample);

      // LOO
      var kmax = 10;
      StatUtils.OptimizeLOO(alg, maxK: kmax);
      var optK = alg.K;
      Console.WriteLine("Nearest K Neigbour: optimal k is {0}", optK);
      Console.WriteLine();

      // Margins
      Console.WriteLine("Margins:");
      calculateMargin(alg);
      Console.WriteLine();

      //Error distribution
      Console.WriteLine("Errors:");
      for (int k = 2; k < kmax; k++)
      {
        alg.K = k;
        var errors = alg.GetErrors(Data.Data, 0, true);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        Console.WriteLine("{0}:\t{1} of {2}\t({3}%) {4}", k, ec, dc, pct, k == optK ? "<-LOO optimal" : string.Empty);
      }
      Console.WriteLine();

      Visualizer.Run(alg);
    }


    private void doBayesianParzenAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var kernel = new GaussianKernel();
      var alg = new BayesianParzenAlgorithm(metric, kernel, 1.0F);
      alg.Train(Data.TrainingSample);

      // LOO
      var hmin = 0.01D;
      var hmax = 5.0D;
      var step = 0.05D;
      StatUtils.OptimizeLOO(alg, hmin, hmax, step);
      var optH = alg.H;
      Console.WriteLine("Bayesian: optimal h is {0}", optH);
      Console.WriteLine();

      // Margins
      Console.WriteLine("Margins:");
      calculateMargin(alg);
      Console.WriteLine();

      //Error distribution
      var message = string.Empty;
      Console.WriteLine("Errors:");
      for (double h1 = hmin; h1 <= hmax; h1 = Math.Round(h1+step, 8))
      {
        var h = h1;
        if (h <= optH && h + step > optH) h = optH;

        alg.H = h;
        var errors = alg.GetErrors(Data.Data, 0, true);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        var mes = string.Format("{0}:\t{1} of {2}\t({3}%) {4}", Math.Round(h, 2), ec, dc, pct, h == optH ? "<-LOO optimal" : string.Empty);
        Console.WriteLine(mes);

        if (h==optH) message = mes;
      }
      Console.WriteLine();
      Console.WriteLine("-----------------------------------------");
      Console.WriteLine("Bayesian: optimal h is {0}", optH);
      Console.WriteLine(message);

      alg.H = optH;
      Visualizer.Run(alg);
    }

    private void doNaiveBayesianKernelAlgorithmTest()
    {
      var kernel = new GaussianKernel();
      var alg = new NaiveBayesianKernelAlgorithm(kernel);
      alg.Train(Data.TrainingSample);

      // LOO
      var hmin = 0.01D;
      var hmax = 5.0D;
      var step = 0.05D;
      StatUtils.OptimizeLOO(alg, hmin, hmax, step);
      var optH = alg.H;
      Console.WriteLine("Naive Bayesian: optimal h is {0}", optH);
      Console.WriteLine();

      // Margins
      Console.WriteLine("Margins:");
      calculateMargin(alg);
      Console.WriteLine();

      //Error distribution
      var message = string.Empty;
      Console.WriteLine("Errors:");
      for (double h1 = hmin; h1 < hmax; h1 += step)
      {
        var h = h1;
        if (h <= optH && h + step > optH) h = optH;

        alg.H = h;
        var errors = alg.GetErrors(Data.Data, 0, true);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        var mes = string.Format("{0}:\t{1} of {2}\t({3}%) {4}", Math.Round(h, 2), ec, dc, pct, h == optH ? "<-LOO optimal" : string.Empty);
        Console.WriteLine(mes);

        if (h==optH) message = mes;
      }
      Console.WriteLine();
      Console.WriteLine("-----------------------------------------");
      Console.WriteLine("Bayesian: optimal h is {0}", optH);
      Console.WriteLine(message);

      alg.H = optH;
      Visualizer.Run(alg);
    }

    private void doNaiveBayesianGeneralAlgorithmTest()
    {
      var distr = new NormalDistribution { UseSigmaMinThreshold=true, SigmaMinThreshold=0.00000001D };
      var alg = new NaiveBayesianGeneralAlgorithm<NormalDistribution, NormalDistribution.Parameters>(distr);
      alg.Train(Data.TrainingSample);

      //Errors
      Console.WriteLine("Errors:");

      var errors = alg.GetErrors(Data.Data, 0, false);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      var mes = string.Format("\t{0} of {1}\t({2}%)", ec, dc, pct);
      Console.WriteLine(mes);

      Visualizer.Run(alg);
    }

    private void doBayesianKernelAlgorithmTest()
    {
      var kernel = new GaussianKernel();
      var alg = new BayesianKernelAlgorithm(kernel);
      alg.Train(Data.TrainingSample);

      // LOO
      var hmin = 0.01D;
      var hmax = 2.0D;
      var step = 0.05D;
      StatUtils.OptimizeLOO(alg, hmin, hmax, step);
      var optH = alg.H;
      Console.WriteLine("Naive Bayesian: optimal h is {0}", optH);
      Console.WriteLine();

      // Margins
      Console.WriteLine("Margins:");
      calculateMargin(alg);
      Console.WriteLine();

      //Error distribution
      var message = string.Empty;
      Console.WriteLine("Errors:");
      for (double h1 = hmin; h1 < hmax; h1 += step)
      {
        var h = h1;
        if (h <= optH && h + step > optH) h = optH;

        alg.H = h;
        var errors = alg.GetErrors(Data.Data, 0, true);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        var mes = string.Format("{0}:\t{1} of {2}\t({3}%) {4}", Math.Round(h, 2), ec, dc, pct, h == optH ? "<-LOO optimal" : string.Empty);
        Console.WriteLine(mes);

        if (h==optH) message = mes;
      }
      Console.WriteLine();
      Console.WriteLine("-----------------------------------------");
      Console.WriteLine("Bayesian: optimal h is {0}", optH);
      Console.WriteLine(message);

      alg.H = optH;
      Visualizer.Run(alg);
    }

    private void doQuadraticDiscriminantAlgorithmTest()
    {
      var alg = new QuadraticDiscriminantAlgorithm();
      alg.Train(Data.TrainingSample);
      Console.WriteLine("Training finished");

      var errors = alg.GetErrors(Data.Data, 0, true);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      var mes = string.Format("{0} of {1}\t({2}%)", ec, dc, pct);
      Console.WriteLine(mes);

      Visualizer.Run(alg);
    }


    private void doParzenFixedAlgorithmTest()
    {
      var timer = new System.Diagnostics.Stopwatch();
      timer.Start();

      var metric = new EuclideanMetric();
      var kernel = new GaussianKernel();
      var alg = new ParzenFixedAlgorithm(metric, kernel, 1.0F);
      alg.Train(Data.TrainingSample);

      // LOO
       StatUtils.OptimizeLOO(alg, 0.1F, 20.0F, 0.2F);
      var optH = alg.H;
      Console.WriteLine("Parzen Fixed: optimal h is {0}", optH);
      Console.WriteLine();

      // Margins
      Console.WriteLine("Margins:");
      calculateMargin(alg);
      Console.WriteLine();

      //var x = algorithm.Classify(new Point(new double[] { -3, 0 }));

      //Error distribution
      Console.WriteLine("Errors:");
      var step = 0.1F;
      for (double h1 = step; h1 < 5; h1 += step)
      {
        var h = h1;
        if (h <= optH && h + step > optH) h = optH;

        alg.H = h;
        var errors = alg.GetErrors(Data.Data, 0, true);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        Console.WriteLine("{0}:\t{1} of {2}\t({3}%) {4}", Math.Round(h, 2), ec, dc, pct, h == optH ? "<-LOO optimal" : string.Empty);
      }
      Console.WriteLine();

      Visualizer.Run(alg);

      timer.Stop();
      Console.WriteLine(timer.ElapsedMilliseconds/1000.0F);
    }

    private void doPotentialFixedAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var kernel = new GaussianKernel();

      var eqps = new PotentialFunctionAlgorithm.KernelEquipment[Data.TrainingSample.Count];
      for (int i=0; i<Data.TrainingSample.Count; i++)
        eqps[i] = new PotentialFunctionAlgorithm.KernelEquipment(1.0F, 1.5F);
      var alg = new PotentialFunctionAlgorithm(metric, kernel, eqps: eqps);
      alg.Train(Data.TrainingSample);

      Console.WriteLine("Margin:");
      calculateMargin(alg);

      outputError(alg);

      Visualizer.Run(alg);
    }

    private void doDecisionTreeAlgorithmTest()
    {
      var alg = new DecisionTreeID3Algorithm<double[]>();
      alg.Patterns = getSimpleLogicPatterns();
      alg.Informativity = new DonskoyIndex<double[]>();
      alg.Train(Data.TrainingSample);

      outputError(alg);

      Visualizer.Run(alg);
    }

    #endregion

    #region NN

    private BackpropAlgorithm createBPAlg()
    {
      var net = NetworkFactory.CreateFullyConnectedNetwork(new[] { 6, 32, 32, 2 }, Activation.Logistic(1));
      //net[0].DropoutRate = 0.1D;
      net.IsTraining = true;

      var alg = new BackpropAlgorithm(net);
      alg.EpochCount = 6000;
      alg.LearningRate = 0.001D;
      alg.BatchSize = 10;
      alg.LossFunction = Loss.Euclidean;

      int epoch = 0;
      alg.EpochEndedEvent += (o, e) =>
                             {
                               if (epoch++ % 300 != 0) return;
                               Console.WriteLine("----------------Epoch #: {0}", epoch);
                               Console.WriteLine("L:\t{0}",  alg.ErrorValue);
                               Console.WriteLine("DL:\t{0}", alg.ErrorDelta);
                               Console.WriteLine("DW:\t{0}", alg.Step2);
                             };

      return alg;
    }

    private void doMultilayerNNAlgorithmTest()
    {
      var alg = createBPAlg();

      var now = DateTime.Now;
      alg.Train(GeneralUtils.ClassifiedToRegressionSample(Data.TrainingSample));

      Console.WriteLine("--------- ELAPSED TRAIN ----------" + (DateTime.Now-now).TotalMilliseconds);

      Console.WriteLine("Loss function: " + alg.ErrorValue);
      Console.WriteLine("Step: " + alg.Step2);

      outputError(alg);

      //Visualizer.Run(alg);
    }

    #endregion

    #region CNN

    private ML.DeepMethods.Algorithms.BackpropAlgorithm createCNNAlg_NN_ForTest()
    {
      var cnn = new ConvNet(6, 1) { IsTraining=true };
      cnn.AddLayer(new DenseLayer(15, activation: Activation.Logistic(1)));
      cnn.AddLayer(new DenseLayer(15, activation: Activation.Logistic(1)));
      cnn.AddLayer(new DenseLayer(15, activation: Activation.Logistic(1)));
      //cnn.AddLayer(new MaxPoolingLayer(1, 1));
      //cnn.AddLayer(new _ActivationLayer(Activation.Logistic(1)));
      //cnn.AddLayer(new DropoutLayer(0.1));
      cnn.AddLayer(new FlattenLayer(2, activation: Activation.Logistic(1)));
      //cnn.AddLayer(new _ActivationLayer(Activation.Logistic(1)));
      //cnn.AddLayer(new MaxPoolingLayer(1, 1));

      cnn._Build();
      cnn.RandomizeParameters(0);

      var alg = new ML.DeepMethods.Algorithms.BackpropAlgorithm(cnn);
      alg.EpochCount = 6000;
      alg.LearningRate = 0.01D;
      alg.BatchSize = 1;
      alg.LossFunction = Loss.Euclidean;

      int epoch = 0;
      alg.EpochEndedEvent += (o, e) =>
                             {
                               if (epoch++ % 300 != 0) return;
                               Console.WriteLine("----------------Epoch #: {0}", epoch);
                               Console.WriteLine("L:\t{0}",  alg.LossValue);
                               Console.WriteLine("DL:\t{0}", alg.LossDelta);
                               Console.WriteLine("DW:\t{0}", alg.Step2);
                             };

      return alg;
    }

    private void doCNNAlgorithmTest()
    {
      var alg = createCNNAlg_NN_ForTest();
      alg.Build();
      var sample = sampleTo3D(Data.TrainingSample);

      var now = DateTime.Now;
      alg.Train(GeneralUtils.ClassifiedToRegressionSample(sample));

      Console.WriteLine("--------- ELAPSED TRAIN ----------" + (DateTime.Now-now).TotalMilliseconds);

      Console.WriteLine("Loss function: " + alg.LossValue);
      Console.WriteLine("Step: " + alg.Step2);

      outputError(alg);

      //Visualizer.Run(alg);
    }

    #endregion

    #region auxillary

    private ClassifiedSample<double[][,]> sampleTo3D(ClassifiedSample<double[]> sample)
    {
      var result = new ClassifiedSample<double[][,]>();
      foreach (var obj in sample)
      {
        var data = obj.Key;

        var key = new double[data.Length][,];
        for (int i=0; i<data.Length; i++)
          key[i] = new double[1,1] { { data[i] } };

        result[key] = obj.Value;
      }

      return result;
    }

    private void outputError(ClassificationAlgorithmBase<double[]> alg)
    {
      Console.WriteLine("Errors:");
      var errors = alg.GetErrors(Data.Data, 0, true);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);
    }

    private void outputError(MultiRegressionAlgorithmBase<double[]> alg)
    {
      Console.WriteLine("Errors:");
      var errors = alg.GetErrors(GeneralUtils.ClassifiedToRegressionSample(Data.Data), 0, true);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);
    }

    private void outputError(MultiRegressionAlgorithmBase<double[][,]> alg)
    {
      Console.WriteLine("Errors:");
      var sample = sampleTo3D(Data.Data);
      var errors = alg.GetErrors(GeneralUtils.ClassifiedToRegressionSample(sample), 0, true);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);
    }

    private void outputError(ClassificationAlgorithmBase<double[,,]> alg)
    {
      Console.WriteLine("Errors:");

      var sample = new ClassifiedSample<double[,,]>();
      foreach (var obj in Data.Data)
      {
        var data = obj.Key;
        var key = new double[data.Length, 1, 1];
        for (int i=0; i<data.Length; i++)
          key[i, 0, 0] = data[i];
        sample[key] = obj.Value;
      }

      var errors = alg.GetErrors(sample, 0, true);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);
    }

    private void outputError(ClassificationAlgorithmBase<double[][,]> alg)
    {
      Console.WriteLine("Errors:");

      var sample = new ClassifiedSample<double[][,]>();
      foreach (var obj in Data.Data)
      {
        var data = obj.Key;

        var key = new double[data.Length][,];
        for (int i=0; i<data.Length; i++)
          key[i] = new double[1,1];

        for (int i=0; i<data.Length; i++)
          key[i][0,0] = data[i];
        sample[key] = obj.Value;
      }

      var errors = alg.GetErrors(sample, 0, true);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);
    }

    private void calculateMargin(IGammaAlgorithm<double[]> algorithm)
    {
      var res = StatUtils.CalculateMargins(algorithm);
      foreach (var r in res)
      {
        Console.WriteLine("{0}\t{1}", r.Key, r.Value);
      }
    }

    private IEnumerable<Predicate<double[]>> getSimpleLogicPatterns()
    {
      var sample = Data.TrainingSample;
      var dim = sample.First().Key.Length;
      double step = 0.05F;

      for (int i=0; i<dim; i++)
      {
        var idx = i;
        var min = sample.Min(p => p.Key[idx]);
        var max = sample.Max(p => p.Key[idx]);

        for (double l=min; l<=max; l += step)
        {
          var level = l;
          yield return (p => idx>=p.Length ? true : p[idx]<level);
        }
      }
    }

    #endregion
  }
}
