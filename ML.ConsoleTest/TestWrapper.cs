using System;
using System.Linq;
using System.Collections.Generic;
using ML.Core;
using ML.Core.Metric;
using ML.Core.Kernels;
using ML.Contracts;
using ML.LogicalMethods.Algorithms;
using ML.MetricalMethods.Algorithms;
using ML.Core.Logical;
using ML.NeuralMethods.Algorithms;

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
        //doNearestNeighbourAlgorithmTest();
        //doNearestKNeighboursAlgorithmTest();
        //doParzenFixedAlgorithmTest();
        //doPotentialFixedAlgorithmTest();

        //doDecisionTreeAlgorithmTest();

        doPerceptronAlgorithmTest();
      }
    }

    #region .pvt

    private void doNearestNeighbourAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var alg = new NearestNeighbourAlgorithm(Data.TrainingSample, metric);

      Console.WriteLine("Margin:");
      calculateMargin(alg);

      Console.WriteLine("Errors:");
      var errors = alg.GetErrors(Data.Data);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);

      Visualizer.Run(alg);
    }

    private void doNearestKNeighboursAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var alg = new NearestKNeighboursAlgorithm(Data.TrainingSample, metric, 1);

      // LOO
      alg.Train_LOO();
      var optK = alg.K;
      Console.WriteLine("Nearest K Neigbour: optimal k is {0}", optK);
      Console.WriteLine();

      // Margins
      Console.WriteLine("Margins:");
      calculateMargin(alg);
      Console.WriteLine();

      //Error distribution
      Console.WriteLine("Errors:");
      for (int k = 1; k < 5; k++)
      {
        alg.K = k;
        var errors = alg.GetErrors(Data.Data);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        Console.WriteLine("{0}:\t{1} of {2}\t({3}%) {4}", k, ec, dc, pct, k == optK ? "<-LOO optimal" : string.Empty);
      }
      Console.WriteLine();

      Visualizer.Run(alg);
    }

    private void doParzenFixedAlgorithmTest()
    {
      var timer = new System.Diagnostics.Stopwatch();
      timer.Start();

      var metric = new EuclideanMetric();
      var kernel = new GaussianKernel();
      var alg = new ParzenFixedAlgorithm(Data.TrainingSample, metric, kernel, 1.0F);

      // LOO
      alg.Train_LOO(0.1F, 20.0F, 0.2F);
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
        var errors = alg.GetErrors(Data.Data);
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
      var alg = new PotentialFunctionAlgorithm(Data.TrainingSample, metric, kernel, eqps);

      Console.WriteLine("Margin:");
      calculateMargin(alg);

      outputError(alg);

      Visualizer.Run(alg);
    }

    private void doDecisionTreeAlgorithmTest()
    {
      var patterns = getSimpleLogicPatterns();

      var alg = new DecisionTreeID3Algorithm(Data.TrainingSample);
      var informativity = new DonskoyIndex();
      alg.Train(patterns, informativity);

      outputError(alg);

      Visualizer.Run(alg);
    }

    private void doPerceptronAlgorithmTest()
    {
      var alg = new SingleLayerBackpropAlgorithm(Data.TrainingSample)
      {
        EpochCount = 10000,
        InputDim = 2,
        OutputDim = 3,
        UseBias = true,
        BatchSize = 10,
        Mode = SingleLayerBackpropAlgorithm.TrainingMode.Batch,
        LearningRate = 0.1D,
        ActivationFunction = Registry.ActivationFunctions.Rational(1)
      };
      alg.Train();

      Console.WriteLine(alg.Error);

      outputError(alg);

      Visualizer.Run(alg);
    }

    private void outputError(AlgorithmBase alg)
    {
      Console.WriteLine("Errors:");
      var errors = alg.GetErrors(Data.Data);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);
    }

    private void calculateMargin(IMetricAlgorithm algorithm)
    {
      var res = algorithm.CalculateMargins();
      foreach (var r in res)
      {
        Console.WriteLine("{0}\t{1}", r.Key, r.Value);
      }
    }

    private IEnumerable<Predicate<Point>> getSimpleLogicPatterns()
    {
      var sample = Data.TrainingSample;
      var dim = sample.First().Key.Dimension;
      double step = 0.05F;

      for (int i=0; i<dim; i++)
      {
        var idx = i;
        var min = sample.Min(p => p.Key[idx]);
        var max = sample.Max(p => p.Key[idx]);

        for (double l=min; l<=max; l += step)
        {
          var level = l;
          yield return (p => idx>=p.Dimension ? true : p[idx]<level);
        }
      }
    }

    #endregion
  }
}
