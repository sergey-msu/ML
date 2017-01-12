using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ML.Core;
using ML.Core.Algorithms;
using ML.Core.Metric;
using ML.Core.Kernels;
using ML.Core.Contracts;
using ML.Core.Stats;

namespace ML.ConsoleTest
{
  public class TestWrapper
  {
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
      //doNearestNeighbourAlgorithmTest();
      //doNearestKNeighboursAlgorithmTest();
      doParzenFixedAlgorithmTest();
      //doPotentialFixedAlgorithmTest();
    }

    #region .pvt

    private void doNearestNeighbourAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var algorithm = new NearestNeighbourAlgorithm(Data.TrainingSample, metric);

      Console.WriteLine("Margin:");
      calculateMargin(algorithm);

      Console.WriteLine("Errors:");
      var errors = General.GetErrors(algorithm, Data.Data);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);

      Visualizer.Run(algorithm);
    }

    private void doNearestKNeighboursAlgorithmTest()
    {
      var metric = new EuclideanMetric();

      // LOO
      var res = LOO.For_NearestKNeighboursAlgorithm(Data.TrainingSample, metric);
      Console.WriteLine("Nearest K Neigbour: optimal k is {0}", res.K);
      Console.WriteLine();

      // Margins
      var pars = new NearestKNeighboursAlgorithm.Params(res.K);
      var algorithm = new NearestKNeighboursAlgorithm(Data.TrainingSample, metric, pars);
      calculateMargin(algorithm);
      Console.WriteLine();

      //Error distribution
      Console.WriteLine("Errors:");
      for (int k = 1; k < 5; k++)
      {
        var p = new NearestKNeighboursAlgorithm.Params(k);
        var alg = new NearestKNeighboursAlgorithm(Data.TrainingSample, metric, p);
        var errors = General.GetErrors(alg, Data.Data);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        Console.WriteLine("{0}:\t{1} of {2}\t({3}%) {4}", k, ec, dc, pct, k == res.K ? "<-LOO optimal" : string.Empty);
      }
      Console.WriteLine();

      Visualizer.Run(algorithm);
    }

    private void doParzenFixedAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var kernel = new GaussianKernel();

      // LOO
      var res = LOO.For_ParzenFixedAlgorithm(Data.TrainingSample, metric, kernel, 0.1F, 20.0F);
      Console.WriteLine("Parzen Fixed: optimal h is {0}", res.H);
      Console.WriteLine();

      // Margins
      Console.WriteLine("Margins:");
      var pars = new ParzenFixedAlgorithm.Params(res.H);
      var algorithm = new ParzenFixedAlgorithm(Data.TrainingSample, metric, kernel, pars);
      calculateMargin(algorithm);
      Console.WriteLine();

      //var x = algorithm.Classify(new Point(new float[] { -3, 0 }));

      //Error distribution
      Console.WriteLine("Errors:");
      var step = 0.1F;
      for (float h1 = step; h1 < 5; h1 += step)
      {
        var h = h1;
        if (h <= res.H && h + step > res.H) h = res.H;

        var p = new ParzenFixedAlgorithm.Params(h);
        var alg = new ParzenFixedAlgorithm(Data.TrainingSample, metric, kernel, p);
        var errors = General.GetErrors(alg, Data.Data);
        var ec = errors.Count();
        var dc = Data.Data.Count;
        var pct = Math.Round(100.0F * ec / dc, 2);
        Console.WriteLine("{0}:\t{1} of {2}\t({3}%) {4}", Math.Round(h, 2), ec, dc, pct, h == res.H ? "<-LOO optimal" : string.Empty);
      }
      Console.WriteLine();

      Visualizer.Run(algorithm);
    }

    private void doPotentialFixedAlgorithmTest()
    {
      var metric = new EuclideanMetric();
      var kernel = new GaussianKernel();

      var gammas = new float[Data.TrainingSample.Count];
      var hs = new float[Data.TrainingSample.Count];
      for (int i=0; i<Data.TrainingSample.Count; i++)
      {
        gammas[i] = 1.0F;
        hs[i] = 1.5F;
      }
      var pars = new PotentialFunctionAlgorithm.Params(gammas, hs);
      var algorithm = new PotentialFunctionAlgorithm(Data.TrainingSample, metric, kernel, pars);

      Console.WriteLine("Margin:");
      calculateMargin(algorithm);

      Console.WriteLine("Errors:");
      var errors = General.GetErrors(algorithm, Data.Data);
      var ec = errors.Count();
      var dc = Data.Data.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);

      var x = errors.Any(e => Data.TrainingSample.Any(t => t.Key==e.Point));

      Visualizer.Run(algorithm);
    }


    private void calculateMargin(IAlgorithm algorithm)
    {
      var res = Margin.Calculate(algorithm);

      foreach (var r in res)
      {
        Console.WriteLine("{0}\t{1}", r.Key, r.Value);
      }
    }

    #endregion
  }
}
